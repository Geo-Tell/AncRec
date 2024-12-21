import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import random

def init_dist_pytorch(local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)


    dist.init_process_group(
        backend=backend,
        init_method="env://", #"'tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank

def initiate_environment(config, rank=0):
    '''
    initiate randomness.
    :param config:
    :return:
    '''
    # rank = torch.distributed.get_rank()
    # 问题完美解决！
    torch.manual_seed(config['seed']+rank)
    torch.cuda.manual_seed_all(config['seed']+rank)
    np.random.seed(config['seed']+rank)
    random.seed(config['seed']+rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True



def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size
