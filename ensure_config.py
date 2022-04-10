# note that yaml file include some preset/default config described by name
# 1. ensure config not conflict
# 2. load preset/default config based on yaml file

from metrics import metric_main


class UserError(Exception):
    pass


def isNone(conf):
    if conf == 'none' or conf == 'None' or conf == None:
        return True
    return False


# cfg_specs = {
#         'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
#         'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
#         'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
#         'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
#         'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
#         'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
#     }

augpipe_specs = {
    'blit': dict(xflip=1, rotate90=1, xint=1),
    'geom': dict(scale=1, rotate=1, aniso=1, xfrac=1),
    'color': dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
    'filter': dict(imgfilter=1),
    'noise': dict(noise=1),
    'cutout': dict(cutout=1),
    'bg': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
    'bgc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1,
                hue=1, saturation=1),
    'bgcf': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1,
                 hue=1, saturation=1, imgfilter=1),
    'bgcfn': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                  lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
    'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                   lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
}

resume_specs = {
    'ffhq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
    'ffhq512': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
    'ffhq1024': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
    'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
    'lsundog256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
}


def ensure_config(config):
    if isNone(config['snap']):
        config['snap'] = 50
    assert config['snap'] >= 1
    # new add 2
    config['image_snapshot_ticks'] = config['snap']
    config['network_snapshot_ticks'] = config['snap']

    if isNone(config['metrics']):
        config['metrics'] = ['fid50k_full']
    assert isinstance(config['metrics'], list)
    if not all(metric_main.is_valid_metric(metric) for metric in config['metrics']):
        raise UserError(
            '\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    if isNone(config['random_seed']):
        config['random_seed'] = 0

    if isNone(config['cond']):
        config['cond'] = False

    # new add 4
    config['train_dataset']['dataset']['args']['use_labels'] = False
    config['train_dataset']['dataset']['args']['max_size'] = None
    config['train_dataset']['dataset']['args']['xflip'] = False
    # config resolution must match dataset's resolution
    config['train_dataset']['dataset']['args']['resolution'] = config['resolution']

    if isNone(config['gamma']):
        config['gamma'] = 0.0002 * (config['resolution'] ** 2) / config['train_dataset']['batch_size']

    if isNone(config['ema_kimg']):
        config['ema_kimg'] = config['train_dataset']['batch_size'] * 10 / 32

    # ensure model
    assert config['model-G']['args']['synthesis_kwargs']['channel_base'] == \
           config['model-D']['args']['channel_base']
    assert config['model-G']['args']['synthesis_kwargs']['channel_max'] == \
           config['model-D']['args']['channel_max']
    assert config['model-G']['args']['synthesis_kwargs']['num_fp16_res'] == \
           config['model-D']['args']['num_fp16_res']
    assert config['model-G']['args']['synthesis_kwargs']['conv_clamp'] == \
           config['model-D']['args']['conv_clamp']

    if isNone(config['kimg']):
        config['kimg'] = 25000

    # for aug config
    if isNone(config['aug']):
        config['aug'] = 'ada'

    config['ada_target'] = None
    # new add config['ada_target']
    if config['aug'] == 'ada':
        config['ada_target'] = 0.6
    elif config['aug'] == 'noaug':
        pass
    elif config['aug'] == 'fixed':
        if isNone(config['p']):
            raise UserError('fixed aug requires specifying p')
    else:
        raise UserError('aug mode not support')

    config['augment_p']=0
    if not isNone(config['p']):
        if config['aug'] != 'fixed':
            raise UserError('p can only be specified with aug=fixed')
        if not 0 <= config['p'] <= 1:
            raise UserError('p must between 0 and 1')
        # new add
        config['augment_p'] = config['p']


    if not isNone(config['target']):
        if config['aug'] != 'ada':
            raise UserError('target can only be specified with aug=ada')
        if not 0 <= config['target'] <= 1:
            raise UserError('target must between 0 and 1')
        # new add
        config['ada_target'] = config['target']

    if isNone(config['augpipe']):
        config['augpipe'] = 'bgc'
    else:
        if config['aug'] == 'noaug':
            raise UserError('augpipe cannot be specified with aug=noaug')

    assert config['augpipe'] in augpipe_specs
    if config['aug'] != 'noaug':
        # new add
        config['model-Aug']['args'] = augpipe_specs[config['augpipe']]

    config['resume_pkl']=None
    # new add config['resume_pkl']
    if isNone(config['resume']):
        config['resume'] = 'noresume'
    elif config['resume'] in resume_specs:
        config['resume_pkl'] = resume_specs[config['resume']]
    else:
        config['resume_pkl'] = config['resume']

    # new add, new change
    if config['resume'] != 'noresume':
        config['ada_kimg'] = 100
        config['ema_rampup'] = None

    if isNone(config['fp32']):
        config['fp32'] = False
    if config['fp32']:
        config['model-G']['args']['synthesis_kwargs']['num_fp16_res'] = 0
        config['model-D']['args']['num_fp16_res'] = 0
        config['model-G']['args']['synthesis_kwargs']['conv_clamp'] = 0
        config['model-D']['args']['conv_clamp'] = 0

    if isNone(config['nhwc']):
        config['nhwc'] = False

    if isNone(config['cudnn_benchmark']):
        config['cudnn_benchmark'] = True

    if isNone(config['allow_tf32']):
        config['allow_tf32'] = False

    return config
