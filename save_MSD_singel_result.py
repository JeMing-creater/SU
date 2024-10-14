"""
保存一张分割结果图
"""

import monai
import nibabel
import numpy as np
import torch
import yaml
from easydict import EasyDict
from src.SlimUNETR.SlimUNETR import SlimUNETR
from monai.networks.nets import SwinUNETR

from src.loader import get_MSD_transforms



def get_MSD_singel_result(config, model, image_choose,device='cpu'):
    model.to(device)
    model.eval()
    image_size = config.trainer.image_size.MSD
    base_path = config.trainer.MSD_HepaticVessel + '/imagesTr/'
    label_path = config.trainer.MSD_HepaticVessel + '/labelsTr/'
    inference = monai.inferers.SlidingWindowInferer(roi_size=monai.utils.ensure_tuple_rep(image_size, 3), device=device,
                                                    sw_device=device, overlap=0.5)

    # 加载一张图片，并做数据预处理
    print('load image')
    _, val_transform = get_MSD_transforms(config=config)
    data = val_transform({
        'image': base_path + image_choose + '.nii.gz',
        'label': label_path + image_choose + '.nii.gz'
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
    # img = data['label']
    # 展示经过模型的图片
    print('show model output')
    affine = img.meta['original_affine']
    seg = img[0].cpu()
    seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
    seg_out[seg[0] == 1.0] = 1
    seg_out[seg[1] == 1.0] = 2
    # nibabel.save(seg_out.astype(np.uint8), f'{image_choose}_seg.nii.gz')
    nibabel.save(nibabel.Nifti1Image(seg_out.astype(np.uint8), affine=affine), f'{image_choose}_seg.nii.gz')


if __name__ == '__main__':
    device = 'cuda:2'
    # 加载模型
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    print('load model')
    model = SlimUNETR(**config.slim_unetr.MSD)

    image_choose = 'hepaticvessel_025'
    get_MSD_singel_result(config, model, image_choose=image_choose,device=device)
