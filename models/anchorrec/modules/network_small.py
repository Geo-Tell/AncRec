from torch import nn
from models.network import BaseNetwork
from models.registers import METHODS, MODULES, LOSSES
from net_utils.nn_distance import huber_loss
import torch

@METHODS.register_module
class anchorrec_single_reconstruct(BaseNetwork):
    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        self.output_gt_z_mesh = cfg.config['data']['output_gt_z_mesh']
        self.use_rgb = cfg.config['data'].get('use_rgb', False)
        if (not cfg.config['model']):
            cfg.log_string('No submodule found. Please check the phase name and model definition.')
            raise ModuleNotFoundError('No submodule found. Please check the phase name and model definition.')

        '''load network blocks'''
        for phase_name, net_spec in cfg.config['model'].items():
            method_name = net_spec['method']
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            if phase_name == 'feature_encode' and cfg.config['mode'] != 'train':
                cfg.config['model'][phase_name]['return_each_layer'] = False
            subnet = MODULES.get(method_name)(cfg.config['model'][phase_name], optim_spec)
            self.add_module(phase_name, subnet)

            '''load corresponding loss functions'''
            loss = LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(
                self.cfg, self.cfg.config['model'][phase_name].get('weight', 1))
            setattr(self, phase_name + '_loss', loss)

        '''freeze submodules or not'''
        self.freeze_modules(cfg)

    def forward(self, data, export_shape=False):
        '''
            input:
        '''
        end_points = {} #anchor点过少，不参与训练
        input_points = data['input_points'] # B,Npt,3
        vote_features = data['input_vote_features'] # B,C
        anchor_features = data['input_anchor_features'] # B,C
        proposal_features = torch.cat([vote_features,anchor_features],-1)
        if type(self.feature_encode).__name__ == 'ResnetPointnet':
            input_features = torch.cat(  # B,Npts, C+3
                [input_points, proposal_features.unsqueeze(1).repeat(1, input_points.shape[1], 1)], dim=2)
            latent_features = self.feature_encode(input_features)
        elif type(self.feature_encode).__name__ == 'ResnetPointnet2':
            latent_features = self.feature_encode(input_points, proposal_features.unsqueeze(1).repeat(1, input_points.shape[1], 1))
        elif type(self.feature_encode).__name__ == 'PCTransformer_small':
            latent_features = self.feature_encode(input_points) #, proposal_features)

        end_points['pred_zs'] = latent_features
        # end_points['pred_zs_pc'] = pc_feature
        # end_points['pred_zs_img'] = img_feature
        # end_points['objectness'] = objectness
        return end_points

    def generate(self, data):
        input_points = data['input_points']  # B,Npt,3
        vote_features = data['input_vote_features']  # B,C
        anchor_features = data['input_anchor_features']  # B,C
        proposal_features = torch.cat([vote_features, anchor_features], -1)
        if self.use_rgb and 'input_rgb_features' in data:
            rgb_features = data['input_rgb_features']  # B,C
            proposal_features = torch.cat([proposal_features, rgb_features], -1)
        if type(self.feature_encode).__name__ == 'ResnetPointnet':
            input_features = torch.cat(  # B,Npts, C+3
                [input_points, proposal_features.unsqueeze(1).repeat(1, input_points.shape[1], 1)], dim=2)
            latent_features = self.feature_encode(input_features)
        elif type(self.feature_encode).__name__ == 'PCTransformer_small':
            latent_features = self.feature_encode(input_points) #, proposal_features)

        '''only generate first 4 instances for acceleration. Full pipeline testing is in network.py '''
        if latent_features.shape[0] > 4:
            latent_features = latent_features[:4, :]
        if not self.output_gt_z_mesh:
            end_points = self.completion.generate(latent_features)
            end_points['pred_zs'] = latent_features
            return end_points

        gt_zs = data['zs']
        # gt_planes = data['plane_ms']
        if gt_zs.shape[0] > 4:
            gt_zs = data['zs'][:4, :]  # 只生成前4个
        gt_valid_mask = gt_zs.sum(-1) != 0
        gt_valid_zs = gt_zs[gt_valid_mask]
        end_points = self.completion.generate(latent_features, gt_z_vector=gt_valid_zs)
        end_points['pred_zs'] = latent_features
        gt_meshes = end_points['gt_meshes']
        if gt_valid_mask.sum() < gt_zs.shape[0]:
            output_gt_meshes = []
            count = 0
            for i in range(gt_zs.shape[0]):
                if gt_valid_mask[i]:
                    output_gt_meshes.append(gt_meshes[count])
                    count += 1
                else:
                    output_gt_meshes.append([])
        end_points['gt_meshes'] = output_gt_meshes
        return end_points

    def getLatentC(self, data):
        input_features = data['input_features']
        c = self.completion.encoder(input_features)
        return c

    def loss(self, est_data, gt_data):
        loss = {}
        pred_zs = est_data['pred_zs']
        gt_zs = gt_data['zs']
        z_loss = huber_loss(pred_zs - gt_zs).sum(dim=1)  # 32
        loss['total'] = z_loss.mean()
        return loss
