# Utility functions during training and testing.
# author: ynie
# date: Feb, 2020

import os
import urllib
import torch
import torch.nn as nn
from torch.utils import model_zoo
from models.registers import METHODS
import sys
from models import method_paths
from torch.utils.tensorboard import SummaryWriter


class CheckpointIO(object):
    '''
    load, save, resume network weights.
    '''
    def __init__(self, cfg, **kwargs):
        '''
        initialize model and optimizer.
        :param cfg: configuration file
        :param kwargs: model, optimizer and other specs.
        '''
        self.cfg = cfg
        self._module_dict = kwargs
        self._module_dict.update({'epoch': 0, 'min_loss': 1e8,  'min_train_loss': 1e8})
        # self._saved_filename = 'model_min_loss.pth'
        self._saved_filename = 'model_last.pth'

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def saved_filename(self):
        return self._saved_filename

    @staticmethod
    def is_url(url):
        scheme = urllib.parse.urlparse(url).scheme
        return scheme in ('http', 'https')

    def get(self, key):
        return self._module_dict.get(key, None)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self._module_dict.update(kwargs)

    def model_state_to_cpu(self, model_state):
        model_state_cpu = type(model_state)()  # ordered dictsave
        for key, val in model_state.items():
            model_state_cpu[key] = val.cpu()
        return model_state_cpu

    def save(self, suffix=None, **kwargs):
        '''
        save the current module dictionary.
        :param kwargs:
        :return:
        '''
        outdict = kwargs
        for k, v in self._module_dict.items():
            if hasattr(v, 'state_dict'):
                if isinstance(v, torch.nn.parallel.DistributedDataParallel):
                    model_state = self.model_state_to_cpu(v.state_dict())
                else:
                    model_state = v.state_dict()
                outdict[k] = model_state
            else:
                outdict[k] = v

        if not suffix:
            filename = self.saved_filename
        else:
            filename = self.saved_filename.replace('last', suffix)

        torch.save(outdict, os.path.join(self.cfg.config['log']['path'], filename))

    def load(self, filename, *domain):
        '''
        load a module dictionary from local file or url.
        :param filename (str): name of saved module dictionary
        :return:
        '''

        if self.is_url(filename):
            return self.load_url(filename, *domain)
        else:
            return self.load_file(filename, *domain)

    def parse_checkpoint(self):
        '''
        check if resume or finetune from existing checkpoint.
        :return:
        '''
        if self.cfg.config['resume']:
            # resume everything including net weights, optimizer, last epoch, last loss.
            self.cfg.log_string('Begin to resume from the last checkpoint.')
            self.resume()
        elif self.cfg.config['finetune']:
            # only load net weights.
            self.cfg.log_string('Begin to finetune from the existing weight.')
            self.finetune()
        else:
            self.cfg.log_string('Begin to train from scratch.')

    def finetune(self):
        '''
        finetune fron existing checkpoint
        :return:
        '''
        if isinstance(self.cfg.config['weight'], str):
            weight_paths = [self.cfg.config['weight']]
        else:
            weight_paths = self.cfg.config['weight']

        for weight_path in weight_paths:
            if not os.path.exists(weight_path):
                self.cfg.log_string('Warning: finetune failed: the weight path %s is invalid. Begin to train from scratch.' % (weight_path))
            else:
                self.load(weight_path, 'net')
                self.cfg.log_string('Weights for finetuning loaded.')


    def load_bsp_decoder(self, load_module =['decoder','generator'], load_encoder = False):
        '''
        finetune fron existing checkpoint
        :return:
        '''
        weight_path = self.cfg.config['bsp_weight']
        if not os.path.exists(weight_path):
            self.cfg.log_string('Warning: bsp weight loaded failure: the weight path %s is invalid.' % (weight_path))
        else:
            self.cfg.log_string('Loading bsp checkpoint from %s to %s.' % (weight_path, 'CPU' if self.cfg.dist else 'GPU'))
            loc_type = torch.device('cpu') if self.cfg.dist else None

            checkpoint = torch.load(weight_path, map_location=loc_type)
            self._module_dict['net'].module.load_weight_from_others(checkpoint, new_prefix='completion',
                                                                    old_prefix =None, load_module = load_module)
            if load_encoder:
                self._module_dict['net'].module.load_weight_from_others(checkpoint, new_prefix='feature_encode',
                                                                        old_prefix =None, load_module=['encoder'])
            self.cfg.log_string('BSP weight loaded.')


    def resume(self):
        '''
        resume the lastest checkpoint
        :return:
        '''
        resumeFromWeight = False
        if 'weight' in self.cfg.config:
            resumeFromWeight = True
            weight_paths = self.cfg.config['weight']
            if isinstance(weight_paths, list):
                weight_paths = weight_paths[0]

        if resumeFromWeight and os.path.exists(weight_paths):
            self.cfg.log_string('resume from weight path %s'%weight_paths)
            self.load(weight_paths)
        else:
            checkpoint_root = os.path.dirname(self.cfg.save_path)
            saved_log_paths = os.listdir(checkpoint_root)
            saved_log_paths.sort(reverse=True)

            for last_path in saved_log_paths:
                last_checkpoint = os.path.join(checkpoint_root, last_path, self.saved_filename)
                if not os.path.exists(last_checkpoint):
                    continue
                else:
                    self.load(last_checkpoint)
                    self.cfg.log_string('Last checkpoint resumed.')
                    return

        # self.cfg.log_string('Warning: resume failed: No checkpoint available. Begin to train from scratch.')

    def load_file(self, filename, *domain):
        '''
        load a module dictionary from file.
        :param filename: name of saved module dictionary
        :return:
        '''
        '''如果分布式训练，则先将模型加载到cpu上'''
        to_cpu = self.cfg.dist
        loc_type = torch.device('cpu') if to_cpu else None
        if os.path.exists(filename):
            self.cfg.log_string('Loading checkpoint from %s to %s.' % (filename, 'CPU' if to_cpu else 'GPU'))
            checkpoint = torch.load(filename, map_location=loc_type)
            # self.cfg.log_string('Loading checkpoint from %s.' % (filename))
            # checkpoint = torch.load(filename)
            scalars = self.parse_state_dict(checkpoint, *domain)
            return scalars

        else:
            raise FileExistsError

    def load_url(self, url, *domain):
        '''
        load a module dictionary from url.
        :param url: url to a saved model
        :return:
        '''
        self.cfg.log_string('Loading checkpoint from %s.' % (url))
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict, domain)
        return scalars

    def parse_state_dict(self, checkpoint, *domain):
        '''
        parse state_dict of model and return scalars
        :param checkpoint: state_dict of model
        :return:
        '''
        for key, value in self._module_dict.items():

            # only load specific key names.
            if domain and (key not in domain):
                continue

            if key in checkpoint:
                if hasattr(value, 'load_state_dict'):
                    if key != 'net':
                        value.load_state_dict(checkpoint[key])
                    else:
                        '''load weights module by module'''
                        value.module.load_weight(checkpoint[key])
                else:
                    self._module_dict.update({key: checkpoint[key]})
            else:
                self.cfg.log_string('Warning: Could not find %s in checkpoint!' % key)

        if not domain:
            # remaining weights in state_dict that not found in our models.
            scalars = {k:v for k,v in checkpoint.items() if k not in self._module_dict}
            if scalars:
                self.cfg.log_string('Warning: the remaining modules %s in checkpoint are not found in our current setting.' % (scalars.keys()))
        else:
            scalars = {}

        return scalars


def load_device(cfg):
    '''
    load device settings
    :param config:
    :return:
    '''
    if cfg.config['device']['use_gpu'] and torch.cuda.is_available():
        cfg.log_string('GPU mode is on.')
        cfg.log_string('GPU Ids: %s used.' % (cfg.config['device']['gpu_ids']))
        torch.cuda.set_device(cfg.LOCAL_RANK)
        print(cfg.LOCAL_RANK)
        return torch.device("cuda", cfg.LOCAL_RANK) #
    else:
        cfg.log_string('CPU mode is on.')
        return torch.device("cpu")

def load_model(cfg, device):
    '''
    load specific network from configuration file
    :param config: configuration file
    :param device: torch.device
    :return:
    '''
    if cfg.config['method'] not in METHODS.module_dict:
        cfg.log_string('The method %s is not defined, please check the correct name.' % (cfg.config['method']))
        cfg.log_string('Exit now.')
        sys.exit(0)

    model = METHODS.get(cfg.config['method'])(cfg)
    model.train()

    if cfg.dist:
        # model.cuda()
        # model.to(cfg.LOCAL_RANK)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        # local_rank = torch.distributed.get_rank()
        # print("load model: %s"%cfg.LOCAL_RANK)
        return nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK% torch.cuda.device_count()]) #, output_device=cfg.LOCAL_RANK) #,find_unused_parameters=True,
    else:
        return nn.DataParallel(model).to(device)

def load_trainer(cfg, net, optimizer, device):
    '''
    load trainer for training and validation
    :param cfg: configuration file
    :param net: nn.Module network
    :param optimizer: torch.optim
    :param device: torch.device
    :return:
    '''
    from models import anchorrec
    trainer = anchorrec.config.get_trainer(cfg=cfg,net=net,optimizer=optimizer,device=device)
    return trainer

def load_tester(cfg, net, device):
    '''
    load tester for testing
    :param cfg: configuration file
    :param net: nn.Module network
    :param device: torch.device
    :return:
    '''
    from models import anchorrec
    tester =  anchorrec.config.get_tester(cfg=cfg,net=net,device=device)
    return tester

def load_dataloader(cfg, mode):
    '''
    load dataloader
    :param cfg: configuration file.
    :param mode: 'train', 'val' or 'test'.
    :return:
    '''
    from models import anchorrec
    dataset, dataloader, sampler = anchorrec.config.get_dataloader(cfg=cfg,mode=mode)
    return dataset, dataloader, sampler

class AverageMeter(object):
    '''
    Computes ans stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val # current value
        if not isinstance(val, list):
            self.sum += val * n # accumulated sum, n = batch_size
            self.count += n # accumulated count
        else:
            self.sum += sum(val)
            self.count += len(val)
        self.avg = self.sum / self.count # current average value

class LossRecorder(object):
    def __init__(self, batch_size=1):
        '''
        Log loss data
        :param config: configuration file.
        :param phase: train, validation or test.
        '''
        self._batch_size = batch_size
        self._loss_recorder = {}

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def loss_recorder(self):
        return self._loss_recorder

    def update_loss(self, loss_dict):
        for key, item in loss_dict.items():
            if key not in self._loss_recorder:
                self._loss_recorder[key] = AverageMeter()
            self._loss_recorder[key].update(item, self._batch_size)

class LogBoard(object):
    def __init__(self, pth):
        self.writer = SummaryWriter(pth)
        self.iter = 1

    def update(self, value_dict, step_len, phase):
        n_iter = self.iter * step_len
        for key, item in value_dict.items():
            self.writer.add_scalar(key + '/' + phase, item, n_iter)
        self.iter += 1