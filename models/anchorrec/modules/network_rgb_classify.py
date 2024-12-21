import torch
from models.network import BaseNetwork
from models.registers import METHODS, MODULES
import torch.nn as nn
criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')

'''For RGB classification pretraining'''
@METHODS.register_module
class anchorrec_rgb_classify(BaseNetwork):
    def __init__(self, cfg):
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        for phase_name, net_spec in cfg.config['model'].items():
            method_name = net_spec['method']
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            print(method_name)
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)
        
        rgb_channels = 256
        self.predict_head = nn.Sequential(
            nn.Linear(rgb_channels, rgb_channels),
            nn.BatchNorm1d(rgb_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(rgb_channels, rgb_channels),
            nn.BatchNorm1d(rgb_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(rgb_channels, 8)
        )

        '''freeze submodules or not'''
        self.freeze_modules(cfg)

    def forward(self, data):
        end_points = {}
        input_images = data['img']
        rgb_features = self.img_encoder(input_images) #B,C
        sem_cls_scores = self.predict_head(rgb_features)
        end_points['sem_cls_scores'] = sem_cls_scores
        return end_points


    def loss(self, est_data, gt_data):
        sem_cls_label = gt_data['cls_id'] # select (B,K) from (B,K2)
        sem_cls_loss = criterion_sem_cls(est_data['sem_cls_scores'], sem_cls_label)  # (B,K)
        sem_cls_loss = sem_cls_loss.mean()

        pred_cls = torch.argmax(est_data['sem_cls_scores'], 1)
        accuracy = (pred_cls==sem_cls_label).sum().item()/len(sem_cls_label)

        loss_dict=dict()
        loss_dict['total'] = sem_cls_loss
        loss_dict['accuracy'] = accuracy
        loss_dict['sem_cls_loss'] = sem_cls_loss.item()
        return loss_dict
