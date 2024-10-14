"""
保存一张分割结果图
"""

import monai
import nibabel
import numpy as np
import torch
import yaml
from easydict import EasyDict
from monai.networks.nets import SwinUNETR
from src.SlimUNETR.SlimUNETR import SlimUNETR
from src.loader import get_Brats_transforms

image_path = 'BraTS2021_00657'
base_path = f'/dataset/cv/seg/BRaTS2021/{image_path}/{image_path}'
device = 'cuda:2'
# 加载模型
config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
print('load model')
# model = MbotSegSwinUnetr(**config.swin_unetr)
# model.load_state_dict(torch.load('finetune21-swin/best/pytorch_model.bin', map_location='cpu'))
# model = SwinUNETR(img_size=(128, 128, 128), in_channels=4, out_channels=3, feature_size=48)
model = SlimUNETR(in_channels= 4,
    out_channels= 3,
    embed_dim= 96, # Different embedding_dim for different image size
    embedding_dim= 64, # Br
    channels= [ 24, 48, 60 ],
    blocks= [ 1, 2, 3, 2 ],
    heads= [ 1, 2, 4, 4 ],
    r= [ 4, 2, 2, 1 ],
    distillation= False,
    dropout= 0.3)

from src.utils import load_model_dict
import collections
state_dict = torch.load('model_store/2021Best/best/pytorch_model.bin', map_location=torch.device('cpu'))
new_state_dict = collections.OrderedDict()
print(state_dict.keys())
# print(test_dict.keys())
for key in state_dict.keys():
    new_key = key
    if 'Encoder' in key:
        if 'conv1' in key:
            new_key = new_key.replace('conv1','DWconv1')
            new_key = new_key.replace('downsample','depth_wise')
        if 'conv2' in key:
            new_key = new_key.replace('conv2','DWconv2')
            new_key = new_key.replace('downsample','depth_wise')
        if 'conv3' in key:
            new_key = new_key.replace('conv3','DWconv3')
            new_key = new_key.replace('downsample','depth_wise')
        if 'conv4' in key:
            new_key = new_key.replace('conv4','DWconv4')
            new_key = new_key.replace('downsample','depth_wise')
    if 'Decoder' in key:
        if 'conv' in key:
            new_key = new_key.replace('downsample','transposed')
        if 'conv1' in key:
            new_key = new_key.replace('conv1', 'SegHead')
        if 'conv2' in key:
            new_key = new_key.replace('conv2', 'TSconv3')
        if 'conv3' in key:
            new_key = new_key.replace('conv3', 'TSconv2')
        if 'conv4' in key:
            new_key = new_key.replace('conv4', 'TSconv1')
    if 'block' in key:
        if 'cpe' in key:
            new_key = new_key.replace('conditional_positional_encoding','positional_encoding')
            new_key = new_key.replace('cpe', 'patch')
        if 'mlp' in key:
            new_key = new_key.replace('mlp_layer', 'line_conv')
            new_key = new_key.replace('mlp_act', 'act')
            new_key = new_key.replace('mlp', 'LineConv')
        if 'LocalAgg' in key:
            new_key = new_key.replace('bn','bn1')
            new_key = new_key.replace('pointwise_prenorm_1', 'bn2')
            new_key = new_key.replace('LocalAgg', 'LocalRC')
        if 'GlobalSparseAttention' in key:
            new_key = new_key.replace('GlobalSparseAttention', 'GlobalST')
        if 'LocalPropagation' in key:
            new_key = new_key.replace('local_prop','conv_trans')
            new_key = new_key.replace('proj', 'pointwise_conv')
            new_key = new_key.replace('LocalPropagation', 'LocalRD')
    new_state_dict[new_key] = state_dict[key]
model.load_state_dict(new_state_dict)
print('加载模型成功')
# model.load_state_dict(torch.load('model_store/2021Best/best/pytorch_model.bin', map_location='cpu'))

model.to(device)
image_size = 128
inference = monai.inferers.SlidingWindowInferer(roi_size=monai.utils.ensure_tuple_rep(image_size, 3), device=device, sw_device=device, overlap=0.5)

# 加载一张图片，并做数据预处理
print('load image')
_, val_transform = get_Brats_transforms(image_size, is2019=config.trainer.is_brats2019)
data = val_transform({
    'image': [
        image_path + '_flair.nii.gz', image_path + '_t1.nii.gz',
        image_path + '_t1ce.nii.gz', image_path + '_t2.nii.gz'],
    'label': image_path + '_seg.nii.gz'
})

data['image'] = torch.unsqueeze(data['image'], dim=0).to(device)
data['label'] = torch.unsqueeze(data['label'], dim=0).to(device)

print('model inference')
with torch.no_grad():
    img = inference(data['image'], model)
post_trans = monai.transforms.Compose([
    monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
])
img = post_trans(img)
# 展示经过模型的图片
print('show model output')
affine = img.meta['original_affine']
seg = img[0].cpu()
seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
seg_out[seg[1] == 1] = 2
seg_out[seg[0] == 1] = 1
seg_out[seg[2] == 1] = 4

# fig = plt.figure()
# matshow3d(volume=img, fig=fig, title="input image", figsize=(100, 100), every_n=10, frame_dim=-1, show=False, cmap="gray", )
# plt.savefig('model.png')
nibabel.save(nibabel.Nifti1Image(seg_out.astype(np.uint8), affine), f'{image_path}_seg.nii.gz')
# saver = SaveImage(output_dir="./output", output_ext=".nii.gz", output_postfix="seg")
# tc = img[0][0]
# tc.meta['filename_or_obj'] = tc.meta['filename_or_obj'].replace('flair', 'tc')
# saver(tc)
# wt = img[0][1]
# wt.meta['filename_or_obj'] = wt.meta['filename_or_obj'].replace('flair', 'wt')
# saver(wt)
# et = img[0][2]
# et.meta['filename_or_obj'] = et.meta['filename_or_obj'].replace('flair', 'et')
# saver(et)
# sys.exit()
