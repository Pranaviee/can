import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_contrastive', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_noise', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    weight_mae = args.weight_mae
    weight_simclr = args.weight_simclr
    weight_noise = args.weight_noise

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = torch.cat([samples[0], samples[1]], dim=0) # contrastive hack
        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, loss_contrastive, loss_noise, _, _ = model(samples, mask_ratio=args.mask_ratio)

 # only reconstruction (MAE) + contrastive loss
        if loss_noise is not None:
            loss_recon = weight_noise * loss_noise + (1 - weight_noise) * loss
            loss_noise_value = loss_noise.item()
        else:
            loss_recon = loss  # No noise loss, use main loss only
            loss_noise_value = 0.
        loss = weight_mae * loss_recon + weight_simclr * loss_contrastive
        loss_value = loss.item()
        
        loss_contrastive_value = loss_contrastive.item()
        

        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_contrastive=loss_contrastive_value)
        metric_logger.update(loss_noise=loss_noise_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_contrastive_value_reduce = misc.all_reduce_mean(loss_contrastive_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}