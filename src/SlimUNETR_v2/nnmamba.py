# nnMamba: 3D Biomedical Image Segmentation, Classification and Landmark Detection with State Space Model
# https://arxiv.org/pdf/2402.03526
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

import time
import math
from einops import rearrange, repeat
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Ourmamba(nn.Module):
    def __init__(self, d_model, d_state= 16, d_conv = 4, expand = 2, num_slices=None, device='cuda'):
        super().__init__()
        dim = d_model
        factory_kwargs = {"device": device, "dtype": None}
        self.activation = "silu"
        self.act = nn.SiLU()
        self.d_model = dim
        self.expand = expand
        self.d_conv = d_conv
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_model / 16) 
        self.d_inner = int(self.expand * self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
    
    def forward(self, x_norm):
        batch, seqlen, dim = x_norm.shape
        conv_state, ssm_state = None, None
        xz = rearrange(
            self.in_proj.weight @ rearrange(x_norm, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)
        # assert self.activation in ["silu", "swish"]
        x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
        )
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
        y = rearrange(y, "b d l -> b l d")
        outm = self.out_proj(y)
        return outm

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, mamba_layer=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.mamba_layer = mamba_layer
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.mamba_layer is not None:
            global_att = self.mamba_layer(x)
            out += global_att
        if self.downsample is not None:
            # if self.mamba_layer is not None:
            #     global_att = self.mamba_layer(x)
            #     identity = self.downsample(x+global_att)
            # else:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_res_layer(inplanes, planes, blocks, stride=1, mamba_layer=None):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes, mamba_layer=mamba_layer))

    return nn.Sequential(*layers)


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.nin = conv1x1(dim, dim)
        self.nin2 = conv1x1(dim, dim)
        self.norm2 = nn.BatchNorm3d(dim)  # LayerNorm
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.norm = nn.BatchNorm3d(dim)  # LayerNorm
        self.relu = nn.ReLU(inplace=True)
        self.mamba = Ourmamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    def forward(self, x):
        B, C = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x)
        x = self.relu(x)
        act_x = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        # x_mamba = self.mamba(x_flat)

        x_flip_l = torch.flip(x_flat, dims=[2])
        x_flip_c = torch.flip(x_flat, dims=[1])
        x_flip_lc = torch.flip(x_flat, dims=[1, 2])
        x_ori = self.mamba(x_flat)
        x_mamba_l = self.mamba(x_flip_l)
        x_mamba_c = self.mamba(x_flip_c)
        x_mamba_lc = self.mamba(x_flip_lc)
        x_ori_l = torch.flip(x_mamba_l, dims=[2])
        x_ori_c = torch.flip(x_mamba_c, dims=[1])
        x_ori_lc = torch.flip(x_mamba_lc, dims=[1, 2])
        x_mamba = (x_ori + x_ori_l + x_ori_c + x_ori_lc) / 4

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        # act_x = self.relu3(x)
        out += act_x
        out = self.nin2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=int(kernel_size / 2),
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class SingleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class Attentionlayer(nn.Module):
    def __init__(self, dim, r=16, act="relu"):
        super(Attentionlayer, self).__init__()
        self.layer1 = nn.Linear(dim, int(dim // r))
        self.layer2 = nn.Linear(int(dim // r), dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        att = self.sigmoid(self.layer2(self.relu(self.layer1(inp))))
        return att.unsqueeze(-1)


class nnMamba(nn.Module):
    def __init__(
        self,
        in_ch=1,
        number_classes=6,
        channels=32,
        blocks=3,
    ):
        super(nnMamba, self).__init__()
        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        # self.mamba_layer_stem = MambaLayer(channels)
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.att1 = Attentionlayer(channels)
        self.layer1 = make_res_layer(
            channels,
            channels * 2,
            blocks,
            stride=2,
            mamba_layer=MambaLayer(channels * 2),
        )

        self.att2 = Attentionlayer(channels * 2)
        self.layer2 = make_res_layer(
            channels * 2,
            channels * 4,
            blocks,
            stride=2,
            mamba_layer=MambaLayer(channels * 4),
        )
        # self.mamba_layer_2 = MambaLayer(channels*4)

        self.att3 = Attentionlayer(channels * 4)
        self.layer3 = make_res_layer(
            channels * 4,
            channels * 8,
            blocks,
            stride=2,
            mamba_layer=MambaLayer(channels * 8),
        )
        # self.mamba_layer_3 = MambaLayer(channels*8)

        self.up5 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv8 = DoubleConv(channels, number_classes)

    def forward(self, x):
        c1 = self.in_conv(x)
        scale_f1 = self.att1(
            self.pooling(c1).reshape(c1.shape[0], c1.shape[1])
        ).reshape(c1.shape[0], c1.shape[1], 1, 1, 1)
        # c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1)
        # c2_s = self.mamba_layer_1(c2) + c2
        scale_f2 = self.att2(
            self.pooling(c2).reshape(c2.shape[0], c2.shape[1])
        ).reshape(c2.shape[0], c2.shape[1], 1, 1, 1)

        c3 = self.layer2(c2)
        # c3_s = self.mamba_layer_2(c3) + c3
        scale_f3 = self.att3(
            self.pooling(c3).reshape(c3.shape[0], c3.shape[1])
        ).reshape(c3.shape[0], c3.shape[1], 1, 1, 1)
        c4 = self.layer3(c3)
        # c4_s = self.mamba_layer_3(c4) + c4

        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3 * scale_f3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2 * scale_f2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1 * scale_f1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c8 = self.conv8(up_8)
        return c8

def test_weight(model, x):
    for i in range(0, 3):
        _ = model(x)
    start_time = time.time()
    output = model(x)
    end_time = time.time()
    need_time = end_time - start_time
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout


def Unitconversion(flops, params, throughout):
    print('params : {} M'.format(round(params / (1000**2), 2)))
    print('flop : {} G'.format(round(flops / (1000**3), 2)))
    print('throughout: {} FPS'.format(throughout))

if __name__ == "__main__":
    model = nnMamba(4, 3).cuda()
    x = torch.zeros((1, 4, 128, 128, 128)).cuda()
    print(model(x).shape)
    flops, param, throughout = test_weight(model, x)
    Unitconversion(flops, param, throughout)
