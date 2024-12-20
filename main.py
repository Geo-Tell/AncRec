import argparse
from configs.config_utils import CONFIG
import shutil
from net_utils.common_utils import initiate_environment, init_dist_pytorch
import torch
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--config', type=str, default='configs/config_files/ISCNet.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='train', help='train, test or demo.')
    parser.add_argument('--local_rank', type=int, default=3, help='local rank for distributed training')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', default=18888)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = CONFIG(args.config)
    cfg.update_config(args.__dict__)

    if cfg.config['multiple_gpus']:
        total_gpus, local_rank = init_dist_pytorch(args.local_rank)
        cfg.__setattr__('dist',True)
        cfg.__setattr__('LOCAL_RANK', local_rank)
        batch_size = cfg.config[args.mode]['batch_size']
        assert batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        cfg.config['train']['batch_size'] = batch_size // total_gpus
        cfg.config['val']['batch_size'] = batch_size // total_gpus
    else:
        cfg.__setattr__('dist',False)
        cfg.__setattr__('LOCAL_RANK',0)

    initiate_environment(cfg.config, rank = cfg.LOCAL_RANK if cfg.config['multiple_gpus'] else 0)

    # if args.mode == 'test':
    if cfg.LOCAL_RANK == 0:
        shutil.copy(args.config, 'config.yaml')
        cfg.copy_all_code()
        '''Configuration'''
        cfg.log_string('Loading configurations.')
        cfg.log_string(cfg.config)
        cfg.write_config()

    '''Run'''
    if "train" in cfg.config['mode']:
        import train
        train.run(cfg, cfg.config['mode'])

    else:
        import test
        test.run(cfg, cfg.config['mode'])






