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
from src.SlimUNETR.SlimUNETR import SlimUNETR
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, load_pretrain_model, MetricSaver, load_model_dict

best_acc = 0
best_class = []


def train(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
          metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
          post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int, step: int):
    # 训练
    model.train()
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch['image'])
        total_loss = 0
        log = ''
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
        accelerator.log({
            'Train/Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(
            f'Epoch [{epoch}/{config.trainer.num_epochs}][{i + 1}/{len(train_loader)}] Training Loss:{total_loss}',
            flush=True)
        step += 1
    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update({
            f'Train/mean {metric_name}': float(batch_acc.mean()),
            f'Train/Tumors {metric_name}': float(batch_acc[0]),
            f'Train/Vessel {metric_name}': float(batch_acc[1])
        })
    # accelerator.print(f'Warn Up Over!')
    accelerator.log(metric, step=epoch)
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
        # accelerator.print(
        #     f'[{i + 1}/{len(val_loader)}] Validation Loading',
        #     flush=True)
        step += 1
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0]
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc.to(accelerator.device)) / accelerator.num_processes
        metrics[metric_name].reset()
        if metric_name == 'dice_metric':
            metric.update({
                f'Val/mean {metric_name}': float(batch_acc.mean()),
                f'Val/Tumors {metric_name}': float(batch_acc[0]),
                f'Val/Vessel {metric_name}': float(batch_acc[1])
            })
            dice_acc = torch.Tensor([metric['Val/mean dice_metric']]).to(accelerator.device)
            dice_class = batch_acc
        else:
            metric.update({
                f'Val/mean {metric_name}': float(batch_acc.mean()),
                f'Val/Tumors {metric_name}': float(batch_acc[0]),
                f'Val/Vessel {metric_name}': float(batch_acc[1])
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

    accelerator.print('加载模型...')
    model = SlimUNETR(**config.slim_unetr.MSD)
    image_size = config.trainer.image_size.MSD

    accelerator.print('加载数据集...')
    train_loader, val_loader = get_dataloader(config)

    inference = monai.inferers.SlidingWindowInferer(roi_size=ensure_tuple_rep(image_size, 3), overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(include_background=True,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True),
        # 'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=True,
        #                                                      reduction=monai.utils.MetricReduction.MEAN_BATCH,
        #                                                      get_not_nans=False)
    }
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])

    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)

    # 加载预训练模型
    model = load_pretrain_model(f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/new/pytorch_model.bin",
                                model,
                                accelerator)

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler,
                                                                                train_loader, val_loader)

    # # 开始验证
    accelerator.print("Start Training！")
    step = 0
    best_eopch = -1
    val_step = 0
    best_acc = 0
    best_class = []
    for epoch in range(0, config.trainer.num_epochs):
        step = train(model, loss_functions, train_loader,
                     optimizer, scheduler, metrics,
                     post_trans, accelerator, epoch, step)

        dice_acc, dice_class, hd95_acc, hd95_class = val_one_epoch(model, inference, val_loader,
                                                                   metrics, val_step,
                                                                   post_trans, accelerator)
        print(f'Epoch [{epoch}/{config.trainer.num_epochs}] dice acc: {dice_acc} best acc: {best_acc}')
        if dice_acc > best_acc:
            best_acc = dice_acc
            best_class = dice_class
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/new/")
            torch.save(model.state_dict(), f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/new/model.pth")

    # accelerator.print(f"最高acc: {metric_saver.best_acc}")
    accelerator.print(f"dice acc: {best_acc}")
    accelerator.print(f"dice class : {best_class}")
    # accelerator.print(f"hd95 acc: {hd95_acc}")
    # accelerator.print(f"hd95 class : {hd95_class}")
    sys.exit(1)
