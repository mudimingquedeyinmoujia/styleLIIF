import argparse
import os

import time
import psutil
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import copy
import json
import datasets
import dnnlib
import models
import utils
import ensure_config
import numpy as np

from models.styleLoss import StyleGAN2Loss
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from torch_utils import misc
from torch_utils import training_stats
from torchvision.utils import make_grid
from utils import setup_snapshot_image_grid
from utils import save_image_grid


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    log('{} dataset: size={}'.format(tag, len(dataset)))
    log('Image shape: {}'.format(dataset.image_shape))
    log('Label shape: {}'.format(dataset.label_shape))
    # dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    dataset_sampler = misc.InfiniteSampler(dataset=dataset, rank=0, num_replicas=1, seed=config['random_seed'])
    loader = DataLoader(dataset=dataset, batch_size=spec['batch_size'], sampler=dataset_sampler,
                        num_workers=spec['num_workers'], pin_memory=True, prefetch_factor=2)
    return loader, dataset


def make_data_loaders():
    train_loader, dataset = make_data_loader(config.get('train_dataset'), tag='train')
    return train_loader, dataset


def prepare_training(device):
    if config['resume'] is None:
        G = models.make(config['model-G']).train().requires_grad_(False).to(device)
        D = models.make(config['model-D']).train().requires_grad_(False).to(device)
        G_ema = copy.deepcopy(G).eval()  # for evaluate
        cur_tick = 0
    else:
        sv_file = torch.load(config['resume'])
        G = models.make(sv_file['G'], load_sd=True).train().requires_grad_(False).to(device)
        D = models.make(sv_file['D'], load_sd=True).train().requires_grad_(False).to(device)
        G_ema = copy.deepcopy(G).eval()
        G_ema.load_state_dict(sv_file['G']['sd_ema'])
        cur_tick = sv_file['cur_tick']

    # todo: distribute

    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema),
                         ('augment_pipe', None)]:
        if name is not None:
            ddp_modules[name] = module

    loss = StyleGAN2Loss(device=device, **ddp_modules, r1_gamma=config['gamma'])

    phases = []

    # prepare phase and optimizer
    G_opt_kwargs = config['optimizer-G']
    D_opt_kwargs = config['optimizer-D']
    G_reg_interval = config['G_reg_interval']
    D_reg_interval = config['D_reg_interval']
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval),
                                                   ('D', D, D_opt_kwargs, D_reg_interval)]:  # 4 16
        if reg_interval is None:  # [Gboth, Dboth]
            opt = utils.make_optimizer(param_list=module.parameters(), optimizer_spec=opt_kwargs)
            phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
        else:  # Lazy regularization.  [Gmain, Greg, Dmain, Dreg]
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs['args']['lr'] = opt_kwargs['args']['lr'] * mb_ratio
            opt_kwargs['args']['betas'] = [beta ** mb_ratio for beta in opt_kwargs['args']['betas']]
            opt = utils.make_optimizer(param_list=module.parameters(), optimizer_spec=opt_kwargs)
            phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]

    return G, D, G_ema, loss, phases, cur_tick


def main(config_, save_path, device):
    global config, log, writer
    config = ensure_config.ensure_config(config_)
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # performance setting
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    torch.backends.cudnn.benchmark = config['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = config['allow_tf32']
    torch.backends.cudnn.allow_tf32 = config['allow_tf32']
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    start_time = time.time()
    # data preparing
    train_loader, training_set = make_data_loaders()
    training_set_iterator = iter(train_loader)
    # phase_real_img,phase_real_label=next(training_set_iterator)
    G, D, G_ema, loss, phases, cur_tick_ = prepare_training(device)

    # init save
    batch_sz = config['train_dataset']['batch_size']
    grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
    # save_image_grid(images, os.path.join(save_path, 'reals.png'), drange=[0, 255], grid_size=grid_size)
    grid_z = torch.randn([labels.shape[0], G.z_dim]).to(device).split(batch_sz)  # dim 0 = batch_gpu
    grid_c = torch.from_numpy(labels).to(device).split(batch_sz)
    images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
    save_image_grid(images, os.path.join(save_path, 'fakes_init.png'), drange=[-1, 1], grid_size=grid_size)

    # start training
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = 0
    batch_idx = 0
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time

    total_kimg = config['kimg']
    ema_kimg = config['ema_kimg']
    ema_rampup = config['ema_rampup']
    # ada_interval = 4
    kimg_per_tick = 4
    # ada_target = config['ada_target']
    # ada_kimg = config['ada_kimg']
    batch_gpu = batch_size = config['train_dataset']['batch_size']
    image_snapshot_ticks = config['snap']

    while True:
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)  # batchsize//numgpus
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in
                         range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(
                    zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = True  # True
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c,
                                          sync=sync, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        cur_nimg += batch_size
        batch_idx += 1

        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        log(' '.join(fields))

        # save image snapshot result
        if (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(save_path, f'fakes{cur_nimg // 1000:06d}.png'), drange=[-1, 1],
                            grid_size=grid_size)
            model_G_spec = config['model-G']
            model_G_spec['sd'] = G.state_dict()
            model_G_spec['sd_ema'] = G_ema.state_dict()
            model_D_spec = config['model-D']
            model_D_spec['sd'] = D.state_dict()
            sv_file = {
                'G': model_G_spec,
                'D': model_D_spec,
                'cur_tick': cur_tick
            }
            torch.save(sv_file,
                       os.path.join(save_path, 'snapshot-tick-{}-{:06d}kimg.pth'.format(cur_tick, cur_nimg // 1000)))

        # todo: evaluate metrics

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break


if __name__ == '__main__':
    # init yaml / save dir / gpu nums
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train-ffhq/train_ffhq_styleGAN2_01_resu.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default='1400kimg')
    parser.add_argument('--gpu', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.gpu)
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save/exp1', save_name)

    main(config, save_path, device)
