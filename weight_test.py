import torch
import time
import yaml
from torch import nn
from easydict import EasyDict
from src.SlimUNETR.SlimUNETR import SlimUNETR as v1
# from src.SlimUNETR_v2.Slim import SlimUNETR
# from src.SlimUNETR_v2.Mamba_light import SlimUNETR
from src.SlimUNETR_v2.Mamba_light_v2 import SlimUNETR
# from src.SlimUNETR_v2.Mamba_light_v3 import SlimUNETR

def test_weight(model, x):
    for i in range(0, 5):
        _ = model(x)
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
    print('throughout: {} FPS'.format(throughout))

if __name__ == '__main__':
    # 读取配置
    device = "cuda:0"
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    x = torch.rand(1, 4, 128,128,128).to(device)
    # model = SlimUNETR(**config.slim_unetr.BraTS).to(device)
    v1 = v1(in_channels=4, out_channels=3, embed_dim=96,embedding_dim=64, channels=(24, 48, 60),
                        blocks=(1, 2, 3, 4), heads=(1, 2, 4, 4), r=(4, 2, 2, 1), distillation=False,
                        dropout=0.3).to(device)
    # model = SlimUNETR(in_channels=4, out_channels=3, embed_dim=96,embedding_dim=64, channels=(24, 48, 60),
    #                     blocks=(1, 1, 1, 1), heads=(1, 2, 4, 4), r=(4, 2, 2, 1), num_slices_list = (64, 32, 16, 8), distillation=False,
    #                     dropout=0.3).to(device)
    model = SlimUNETR(in_channels=4, out_channels=3, embed_dim=96,embedding_dim=8, channels=(24, 48, 60),
                        blocks=(1, 1, 1, 1), heads=(1, 2, 4, 4), r=(4, 4, 2, 2), num_slices_list = (64, 32, 16, 8), distillation=False,
                        dropout=0.3).to(device)
    
    _= v1(x)
    flops, param, throughout = test_weight(v1, x)
    Unitconversion(flops, param, throughout)
    # =================================================
    _= model(x)
    flops, param, throughout = test_weight(model, x)
    Unitconversion(flops, param, throughout)
