import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange, repeat

from torch.cuda.amp import custom_bwd, custom_fwd
from timm.models.layers import DropPath, to_2tuple, to_3tuple, make_divisible, trunc_normal_
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))
    
# Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=128, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W, self.D = img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]
        self.num_patches = self.H * self.W * self.D
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            # 计算fan_out，注意Conv3d的情况下kernel_size是一个三元组
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            # 使用标准正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果存在偏置，则将其初始化为0
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # B, N, C = x.shape
        # x = x.permute(0, 2, 1).reshape(B, C, H, W, D)
        return x, H, W, D

class Mlp(nn.Module):
    def __init__(self, channels, shallow=False):
        super(Mlp, self).__init__()
        expansion = 4
        self.line_conv_0 = nn.Conv3d(channels, channels * expansion, kernel_size=1, bias=False)
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        self.line_conv_1 = nn.Conv3d(channels * expansion, channels, kernel_size=1, bias=False)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            # 计算fan_out，注意Conv3d的情况下kernel_size是一个三元组
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            # 使用标准正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果存在偏置，则将其初始化为0
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.line_conv_0(x)
        x = self.act(x)
        x = self.line_conv_1(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out

class MambaBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, num_slices=64, drop_path=0., shallow=False):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mamba = MambaLayer(dim, num_slices=num_slices)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Mlp = Mlp(dim, shallow=shallow)
        
    def forward(self, x, H, W, D):
        B = x.shape[0]
        C = x.shape[-1]
        
        ori_x = x
        
        # norm
        x = self.norm1(x)
        
        # Mamba
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        
        # res
        x = ori_x + self.drop_path(x)
        
        ori_x = x
        # norm
        x = self.norm2(x)
        
        # resize
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D)
        ori_x = ori_x.permute(0, 2, 1).reshape(B, C, H, W, D)
        
        # Mlp
        x = self.Mlp(x)
        out = ori_x + self.drop_path(x)
        out = out.flatten(2).transpose(1, 2)
        return out


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=128, in_chans=4, drop_path=0., depths = [3, 4, 6, 3], embed_dims=[64, 128, 256, 512], num_slices_list = [64, 32, 16, 8]):
        super().__init__()
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_chans,
            out_channels=embed_dims[0],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        
        self.block1 = nn.ModuleList([MambaBlock(dim=embed_dims[0],num_slices=num_slices_list[0], drop_path=drop_path, shallow=True) for i in range(depths[0])])
        self.block2 = nn.ModuleList([MambaBlock(dim=embed_dims[1],num_slices=num_slices_list[1], drop_path=drop_path, shallow=True) for i in range(depths[1])])
        self.block3 = nn.ModuleList([MambaBlock(dim=embed_dims[2],num_slices=num_slices_list[2], drop_path=drop_path, shallow=False) for i in range(depths[2])])
        self.block4 = nn.ModuleList([MambaBlock(dim=embed_dims[3],num_slices=num_slices_list[3], drop_path=drop_path, shallow=False) for i in range(depths[3])])
        
        # self.apply(self._init_weights)  
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            # 计算fan_out，注意Conv3d的情况下kernel_size是一个三元组
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            # 使用标准正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果存在偏置，则将其初始化为0
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        B = x.shape[0]
        C = x.shape[-1]
        outs = []
        first = self.encoder1(x)
        
        x, H, W, D = self.patch_embed1(x)
        for _, blk in enumerate(self.block1):
            x = blk(x, H, W, D)
        C = x.shape[-1]
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D).contiguous()   
        
        # first = self.encoder1(x)
        
        outs.append(x)
        
        x, H, W, D = self.patch_embed2(x)
        for _, blk in enumerate(self.block2):
            x = blk(x, H, W, D)
        C = x.shape[-1]
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D).contiguous() 
        outs.append(x)
            
        x, H, W, D = self.patch_embed3(x)
        for _, blk in enumerate(self.block3):
            x = blk(x, H, W, D)
        C = x.shape[-1]
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D).contiguous() 
        outs.append(x)
           
        x, H, W, D = self.patch_embed4(x)
        for _, blk in enumerate(self.block4):
            x = blk(x, H, W, D)
        C = x.shape[-1]
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D).contiguous() 
        outs.append(x)
        
        outs.append(first)
        return outs


class GobleAttention(nn.Module):
    def __init__(self, in_dim=1, out_dim=32, num_slices=8, act=nn.GELU()):
        super(GobleAttention, self).__init__()
        # 调整通道Conv
        self.conv = nn.Conv3d(in_dim, out_dim, 3, 1, 1)   
        # 特征Norm
        self.norm = nn.GroupNorm(out_dim // 2, out_dim)
        # 激活函数
        self.act = act

        # Mamba
        self.mamba = MambaLayer(out_dim, num_slices=num_slices)
        
        # MLP
        self.mlp = Mlp(out_dim)
        
        # self.apply(self._init_weights) 
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            # 计算fan_out，注意Conv3d的情况下kernel_size是一个三元组
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            # 使用标准正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果存在偏置，则将其初始化为0
            if m.bias is not None:
                m.bias.data.zero_()
        
    def forward(self, x):
        # 调整通道
        x = self.conv(x)
        # 特征Norm
        x = self.norm(x)
        # 特征激活
        x = self.act(x)
        # keep input
        identity = x
        
        # Mamba
        B, C, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D).contiguous() 
        x = x + identity
        
        # MLP
        x = self.mlp(x)
        
        return x + identity

class LocalAttention(nn.Module):
    def __init__(self, in_dim=32, out_dim=32):
        super(LocalAttention, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_dim)
        self.pointwise_conv_0 = nn.Conv3d(in_dim, in_dim, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv3d(in_dim, in_dim, padding=1, kernel_size=3, groups=in_dim, bias=False)
        self.bn2 = nn.BatchNorm3d(in_dim)
        self.pointwise_conv_1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, bias=False)
        # self.apply(self._init_weights) 
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            # 计算fan_out，注意Conv3d的情况下kernel_size是一个三元组
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            # 使用标准正态分布初始化权重
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果存在偏置，则将其初始化为0
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.bn1(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pointwise_conv_1(x)
        return x

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, act=nn.ReLU(inplace=True)):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class FeatureBlock(nn.Module):
    def __init__(self, in_dim=3, num_slices=8, shallow=True):
        super(FeatureBlock, self).__init__()
        self.in_dim = in_dim
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        self.gobel_attention = GobleAttention(in_dim=in_dim//2, out_dim=in_dim, act=self.act, num_slices=num_slices)
        self.local_attention = LocalAttention(in_dim=in_dim//2, out_dim=in_dim)
        self.downsample = BasicConv3d(in_dim*2, in_dim, 1, act=self.act)

    def forward(self, x):
        x_0, x_1 = x.chunk(2,dim = 1)
        x_0 = self.gobel_attention(x_0)
        x_1 = self.local_attention(x_1)
        x = torch.cat([x_0, x_1], dim=1)
        x = self.downsample(x)
        return x



class SlimUNETR(nn.Module):
    def __init__(self, img_size=128, in_chans=4, out_chan=3, kernel_size=3, mlp_ratio=4, drop_path=0., depths = [3, 4, 6, 3], out_dim = 32, embed_dims=[64, 128, 256, 512], num_slices_list = [64, 32, 16, 8]):
        super(SlimUNETR, self).__init__()
        self.backbone = PyramidVisionTransformerImpr(img_size=img_size, in_chans=in_chans, drop_path=drop_path, depths = depths, embed_dims=embed_dims, num_slices_list = num_slices_list)
        
        self.Upsample1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dims[3],
            out_channels=embed_dims[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.Upsample2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dims[2],
            out_channels=embed_dims[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.Upsample3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dims[1],
            out_channels=embed_dims[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.Upsample4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=embed_dims[0],
            out_channels=embed_dims[0],
            kernel_size=3,
            upsample_kernel_size=4,
            norm_name="instance",
            res_block=True,
        )
        self.SegHead = UnetOutBlock(spatial_dims=3, in_channels=embed_dims[0], out_channels=out_chan)
        
        self.block4 = FeatureBlock(in_dim=embed_dims[0], num_slices=num_slices_list[1], shallow=True)
        self.block3 = FeatureBlock(in_dim=embed_dims[1], num_slices=num_slices_list[1], shallow=True)
        self.block2 = FeatureBlock(in_dim=embed_dims[2], num_slices=num_slices_list[2],shallow=False)
        self.block1 = FeatureBlock(in_dim=embed_dims[3], num_slices=num_slices_list[3],shallow=False)
    
    def forward(self, x):
        pvt = self.backbone(x)
        c1, c2, c3, c4, init = pvt
        
        c4 = self.block1(c4)
        x = self.Upsample1(c4, c3)
        x = self.block2(x)
        x = self.Upsample2(x, c2)
        x = self.block3(x)
        x = self.Upsample3(x, c1)
        x = self.block4(x)
        x = self.Upsample4(x, init)
        
        result = self.SegHead(x)
        
        return result
    
if __name__ == '__main__':
    device = 'cuda:0'
    x = torch.randn(size=(2, 4, 128, 128, 128)).to(device)
    model = SlimUNETR().to(device)
    print(model(x).shape)