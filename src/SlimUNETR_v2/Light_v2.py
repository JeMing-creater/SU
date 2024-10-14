import math
import time
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


class SlimUNETR(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, embed_dim=512, embedding_dim=64, channels=(64, 128, 256),
                        blocks=(1, 2, 3, 2), heads=(1, 2, 4, 4), r=(4, 2, 2, 1), num_slices_list = (64, 32, 16, 8), distillation=False,
                        dropout=0.3):
        super(SlimUNETR, self).__init__()
        

    def forward(self, x):
        
        return x
    
def test_weight(model, x):
    start_time = time.time()
    output = model(x)
    end_time = time.time()
    need_time = end_time - start_time
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout


# def Unitconversion(flops, params, throughout):
#     print('params : {} K'.format(round(params *1024 / 10000000, 2)))
#     print('flop : {} M'.format(round(flops *1024 / 10000000000, 2)))
#     print('throughout: {}'.format(throughout * 60))

def Unitconversion(flops, params, throughout):
    print('params : {} M'.format(round(params / (1000**2), 2)))
    print('flop : {} G'.format(round(flops / (1000**3), 2)))
    print('throughout: {}'.format(throughout * 60))

if __name__ == '__main__':
    device = 'cuda:2'
    x = torch.randn(size=(1, 4, 128, 128, 128)).to(device)
    model = SlimUNETR().to(device)
    print(model(x).shape)
    flops, param, throughout = test_weight(model, x)
    Unitconversion(flops, param, throughout)
    