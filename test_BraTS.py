import os
import sys
from datetime import datetime
from typing import Dict

import monai
import pytz
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader
# from src.SlimUNETR.SlimUNETR import SlimUNETR
# from src.SlimUNETR_v2.Mamba_light_v2 import SlimUNETR
from src.SlimUNETR_v2.Mamba_light_v8 import SlimUNETR
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, load_pretrain_model, MetricSaver, load_model_dict

best_acc = 0
best_class = []


def warm_up(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
                    post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int, step: int):
    # 训练
    model.train()
    # accelerator.print(f'Start Warn Up!')
    for i, image_batch in enumerate(train_loader):
        # output
        logits = model(image_batch['image'])
        total_loss = 0
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch['label'])
            accelerator.log({'Train/' + name: float(loss)}, step=step)
            total_loss += alpth * loss
        # val_outputs = [post_trans(i) for i in logits]
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch['label'])
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        
        accelerator.log({
            'Train/Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(
            f'Epoch [{epoch}/{config.trainer.num_epochs}][{i + 1}/{len(train_loader)}] Training Loss:{total_loss}',
            flush=True)
        step += 1
        # break
    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate().to(accelerator.device)
        # print(f'b : {batch_acc.device}')
        # print(f'ad : {accelerator.device}')
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update({
            f'Train/mean {metric_name}': float(batch_acc.mean()),
            f'Train/TC {metric_name}': float(batch_acc[0]),
            f'Train/WT {metric_name}': float(batch_acc[1]),
            f'Train/ET {metric_name}': float(batch_acc[2]),
        })
    return step


@torch.no_grad()
def val_one_epoch(model: torch.nn.Module,
                  inference: monai.inferers.Inferer, val_loader: torch.utils.data.DataLoader,
                  metrics: Dict[str, monai.metrics.CumulativeIterationMetric], step: int,
                  post_trans: monai.transforms.Compose, accelerator: Accelerator):
    # 验证
    model.eval()
    dice_acc = 0
    dice_class = []
    hd95_acc = 0
    hd95_class = []
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch['image'], model)
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch['label'])
        step += 1
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate().to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc.to(accelerator.device)) / accelerator.num_processes
        metrics[metric_name].reset()
        if metric_name == 'dice_metric':
            metric.update({
                f'Val/mean {metric_name}': float(batch_acc.mean()),
                f'Val/TC {metric_name}': float(batch_acc[0]),
                f'Val/WT {metric_name}': float(batch_acc[1]),
                f'Val/ET {metric_name}': float(batch_acc[2]),
            })
            dice_acc = torch.Tensor([metric['Val/mean dice_metric']]).to(accelerator.device)
            dice_class = batch_acc
        else:
            metric.update({
                f'Val/mean {metric_name}': float(batch_acc.mean()),
                f'Val/TC {metric_name}': float(batch_acc[0]),
                f'Val/WT {metric_name}': float(batch_acc[1]),
                f'Val/ET {metric_name}': float(batch_acc[2]),
            })
            hd95_acc = torch.Tensor([metric['Val/mean hd95_metric']]).to(accelerator.device)
            hd95_class = batch_acc
    return dice_acc, dice_class, hd95_acc, hd95_class


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print('load model...')
    # model = SlimUNETR(**config.slim_unetr.BraTS)
    # model = SlimUNETR(**config.slim_unetr.Mamba)
    # model = SlimUNETR(in_channels=4, out_channels=3, embed_dim=96,embedding_dim=64, channels=(24, 48, 60),
    #                     blocks=(1, 2, 3, 2), heads=(1, 2, 4, 4), r=(4, 2, 2, 1),num_slices_list = (64, 32, 16, 8), distillation=False,
    #                     dropout=0.3) # 92.57
    # model = SlimUNETR(in_channels=4, out_channels=3, embed_dim=96,embedding_dim=64, channels=(24, 48, 60),
    #                     blocks=(1, 1, 1, 1), heads=(1, 2, 4, 4), r=(4, 2, 2, 1),num_slices_list = (64, 32, 16, 8), distillation=False,
    #                     dropout=0.3) # 93.51
    # model = SlimUNETR(in_channels=4, out_channels=3, embed_dim=96,embedding_dim=8, channels=(24, 48, 60),
    #                     blocks=(1, 1, 1, 1), heads=(1, 2, 4, 4), r=(4, 4, 2, 2), num_slices_list = (64, 32, 16, 8), distillation=False,
    #                     dropout=0.3)
    model = SlimUNETR(in_chans=4, out_chans=3, kernel_sizes=[4, 2, 2, 2], depths=[2, 2, 2, 2], dims=[48, 96, 192, 384], heads=[1, 2, 4, 4], hidden_size=768, num_slices_list = [64, 32, 16, 8],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3])
    
    # model = SSLHead()
    image_size = config.trainer.image_size.BraTS

    accelerator.print('load dataset...')
    train_loader, val_loader = get_dataloader(config)

    inference = monai.inferers.SlidingWindowInferer(roi_size=ensure_tuple_rep(image_size, 3), overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(include_background=True,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=False),
        'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=True,
                                                             reduction=monai.utils.MetricReduction.MEAN_BATCH,
                                                             get_not_nans=False)
    }
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])


    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)
    step = 0
    best_eopch = -1
    val_step = 0



    # from src.utils import load_model_dict
    # import collections
    # state_dict = torch.load(f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin", map_location=torch.device('cpu'))
    # new_state_dict = collections.OrderedDict()
    # print(state_dict.keys())
    # # print(test_dict.keys())
    # for key in state_dict.keys():
    #     new_key = key
    #     if 'Encoder' in key:
    #         if 'conv1' in key:
    #             new_key = new_key.replace('conv1','DWconv1')
    #             new_key = new_key.replace('downsample','depth_wise')
    #         if 'conv2' in key:
    #             new_key = new_key.replace('conv2','DWconv2')
    #             new_key = new_key.replace('downsample','depth_wise')
    #         if 'conv3' in key:
    #             new_key = new_key.replace('conv3','DWconv3')
    #             new_key = new_key.replace('downsample','depth_wise')
    #         if 'conv4' in key:
    #             new_key = new_key.replace('conv4','DWconv4')
    #             new_key = new_key.replace('downsample','depth_wise')
    #     if 'Decoder' in key:
    #         if 'conv' in key:
    #             new_key = new_key.replace('downsample','transposed')
    #         if 'conv1' in key:
    #             new_key = new_key.replace('conv1', 'SegHead')
    #         if 'conv2' in key:
    #             new_key = new_key.replace('conv2', 'TSconv3')
    #         if 'conv3' in key:
    #             new_key = new_key.replace('conv3', 'TSconv2')
    #         if 'conv4' in key:
    #             new_key = new_key.replace('conv4', 'TSconv1')
    #     if 'block' in key:
    #         if 'cpe' in key:
    #             new_key = new_key.replace('conditional_positional_encoding','positional_encoding')
    #             new_key = new_key.replace('cpe', 'patch')
    #         if 'mlp' in key:
    #             new_key = new_key.replace('mlp_layer', 'line_conv')
    #             new_key = new_key.replace('mlp_act', 'act')
    #             new_key = new_key.replace('mlp', 'LineConv')
    #         if 'LocalAgg' in key:
    #             new_key = new_key.replace('bn','bn1')
    #             new_key = new_key.replace('pointwise_prenorm_1', 'bn2')
    #             new_key = new_key.replace('LocalAgg', 'LocalRC')
    #         if 'GlobalSparseAttention' in key:
    #             new_key = new_key.replace('GlobalSparseAttention', 'GlobalST')
    #         if 'LocalPropagation' in key:
    #             new_key = new_key.replace('local_prop','conv_trans')
    #             new_key = new_key.replace('proj', 'pointwise_conv')
    #             new_key = new_key.replace('LocalPropagation', 'LocalRD')
    #     new_state_dict[new_key] = state_dict[key]
    # model.load_state_dict(new_state_dict)
    # print('加载模型成功')


    # 加载预训练模型
    model = load_pretrain_model(f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/new/pytorch_model.bin", model, accelerator)

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler,
                                                                                train_loader, val_loader)

    # # 开始验证
    accelerator.print("Start Val！")

    _ = warm_up(model, loss_functions, train_loader,
                           optimizer, scheduler, metrics,
                           post_trans, accelerator, 0, step)

    dice_acc, dice_class, hd95_acc, hd95_class = val_one_epoch(model, inference, val_loader,
                                                               metrics, val_step,
                                                               post_trans, accelerator)
    accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/new/")
    # accelerator.print(f"最高acc: {metric_saver.best_acc}")
    accelerator.print(f"dice acc: {dice_acc}")
    accelerator.print(f"dice class : {dice_class}")
    accelerator.print(f"hd95 acc: {hd95_acc}")
    accelerator.print(f"hd95 class : {hd95_class}")
    sys.exit(1)
