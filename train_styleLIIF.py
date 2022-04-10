import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import copy

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
    return loader,dataset


def make_data_loaders():
    train_loader,dataset = make_data_loader(config.get('train_dataset'), tag='train')
    return train_loader,dataset


def prepare_training():
    G = models.make(config['model-G']).train().requires_grad_(False).cuda()
    D = models.make(config['model-D']).train().requires_grad_(False).cuda()
    G_ema = copy.deepcopy(G).eval()  # for evaluate

    augment_pipe = None
    ada_stats = None
    if (config['model-Aug']['args'] is not None) and (config['augment_p'] > 0 or config['ada_target'] is not None):
        augment_pipe = models.make(config['model-Aug']).train().requires_grad_(
            False).cuda()  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(config['augment_p']))  # p is a buffer
        if config['ada_target'] is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # todo: resume
    # todo: distribute
    device = torch.device('cuda:0')

    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema),
                         ('augment_pipe', augment_pipe)]:
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
            opt = utils.make_optimizer(param_list=module.parameters(),optimizer_spec=opt_kwargs)
            phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]

    return G, D, G_ema, augment_pipe, ada_stats, loss, phases


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'], batch['cell'])

        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None;
        loss = None

    return train_loss.item()  # 每次迭代的loss做一个平均，作为此次epoch的loss


def main(config_, save_path):
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

    # data preparing
    train_loader,training_set = make_data_loaders()
    training_set_iterator = iter(train_loader)
    # phase_real_img,phase_real_label=next(training_set_iterator)
    G, D, G_ema, augment_pipe, ada_stats, loss, phases = prepare_training()

    # init save
    batch_sz=config['train_dataset']['batch_size']
    grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
    save_image_grid(images, os.path.join(save_path, 'reals.png'), drange=[0, 255], grid_size=grid_size)
    grid_z = torch.randn([labels.shape[0], G.z_dim]).cuda().split(batch_sz)  # dim 0 = batch_gpu
    grid_c = torch.from_numpy(labels).cuda().split(batch_sz)
    images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
    save_image_grid(images, os.path.join(save_path, 'fakes_init.png'), drange=[-1, 1], grid_size=grid_size)


    timer = utils.Timer()


if __name__ == '__main__':
    # init yaml / save dir / gpu nums
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train-ffhq/train_ffhq_styleGAN2_01.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save/exp1', save_name)

    main(config, save_path)
