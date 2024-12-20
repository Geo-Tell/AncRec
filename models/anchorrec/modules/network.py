import torch
from models.network import BaseNetwork
from models.registers import METHODS, MODULES, LOSSES
from net_utils.ap_helper import parse_predictions, parse_groundtruths, assembly_gt_map_cls
from net_utils.quad_ap_helper import parse_quad_groundtruths, parse_quad_predictions
from net_utils.nn_distance import nn_distance_self
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import  QueryFromAnchors_per_proposal
import numpy as np

@METHODS.register_module
class anchorrec(BaseNetwork):
    def __init__(self, cfg):
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        self.mode =  cfg.config['mode']
        self.phase = cfg.config[self.mode]['phase']

        phase_names = []
        print("phase name: %s"%self.phase)
        if self.phase == 'detection':
            phase_names += ['backbone', 'detection']
        elif self.phase == 'quad_detection':
            phase_names += ['backbone', 'quad_detection']
        elif self.phase == 'joint_detection':
            phase_names += ['backbone', 'quad_detection', 'detection']
        elif self.phase == 'prepare_data':
            phase_names += ['backbone', 'detection']
        elif self.phase == 'final_recon':
            phase_names += ['backbone', 'detection','feature_encode','completion']

        if (not cfg.config['model']) or (not phase_names):
            cfg.log_string('No submodule found. Please check the phase name and model definition.')
            raise ModuleNotFoundError('No submodule found. Please check the phase name and model definition.')

        '''load network blocks'''
        for phase_name, net_spec in cfg.config['model'].items():
            if phase_name not in phase_names:
                continue
            method_name = net_spec['method']
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            if phase_name == "completion"or phase_name == "feature_encode":
                subnet = MODULES.get(method_name)(cfg.config['model'][phase_name], optim_spec)
            else:
                subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)

            '''load corresponding loss functions'''
            # 调用 class BaseLoss (in loss.py)
            if 'loss' in self.cfg.config['model'][phase_name]:
                loss = LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(
                    self.cfg, self.cfg.config['model'][phase_name].get('weight', 1))
                # self.cfg.config, self.cfg.config['model'][phase_name].get('weight', 1)))
                setattr(self, phase_name + '_loss', loss)
        '''freeze submodules or not'''
        self.freeze_modules(cfg)

    def generate(self, data):
        points = data['point_clouds']
        end_points = {}
        eval_dict = {}
        parsed_predictions = {}
        with torch.no_grad():
            end_points = self.backbone(points, end_points)
            # -----QUAD DETECTION ------
            if self.phase == 'quad_detection' or self.phase == 'joint_detection':
                end_points = self.quad_detection(end_points)
            # --------- DETECTION ---------
            if self.phase == 'detection' or self.phase == 'joint_detection':
                end_points = self.detection(end_points)

        if self.phase == 'quad_detection' or self.phase == 'joint_detection':
            has_quad_ind = data['use_quad'].bool()
            parsed_predictions, eval_dict = parse_quad_predictions(parsed_predictions, eval_dict, end_points,
                                                                   has_quad_ind, self.cfg)
            eval_dict = parse_quad_groundtruths(eval_dict, data)
            parsed_gts = parse_groundtruths(data, self.cfg.dataset_config)
            eval_dict['batch_gt_map_cls'] = assembly_gt_map_cls(parsed_gts)
            end_points['parsed_gts'] = parsed_gts

        if self.phase == 'detection' or self.phase == 'joint_detection':
            parsed_predictions, eval_dict = parse_predictions(parsed_predictions, eval_dict, end_points, data,
                                                              self.cfg.eval_config)
            parsed_gts = parse_groundtruths(data, self.cfg.dataset_config)
            eval_dict['batch_gt_map_cls'] = assembly_gt_map_cls(parsed_gts)
            end_points['parsed_gts'] = parsed_gts

        elif self.phase == "final_recon":
            end_points = self.generate_input(data)
            parsed_predictions, eval_dict = parse_predictions(parsed_predictions, eval_dict, end_points, data,
                                                              self.cfg.eval_config, return_batch_pred_map_cls=False)
            pred_mask = eval_dict['pred_mask']
            if 'above_floor_mask' in parsed_predictions:
                pred_mask = pred_mask * parsed_predictions['above_floor_mask']
            obj_probs = parsed_predictions['obj_prob']
            dump_thresh = self.cfg.config[self.mode].get('dump_thresh', 0.05)
            sampled_ids = (obj_probs > dump_thresh) * pred_mask  # B,N
            end_points['sampled_ids'] = np.where(sampled_ids[0])[0]
            vote_features = end_points['vote_features']  # B,C,N
            anchor_features = end_points['anchor_features']  # B,C,N
            proposal_features = torch.cat([vote_features, anchor_features], 1).transpose(1,2).contiguous() #B,N,C
            input_points = torch.tensor(end_points['anchor_sampled_pts_list']).to(vote_features.device)  # N,Npt,3

            input_features = torch.cat(  # B,Npts, C+3
                [input_points, proposal_features[0].unsqueeze(1).repeat(1, input_points.shape[1], 1)], dim=2) #N,Npt,c+3
            latent_features = self.feature_encode(input_features)
            sampled_ids = torch.tensor(sampled_ids[0]).to(vote_features.device)
            end_points_meshes = self.completion.generate(latent_features[sampled_ids.bool()])
            end_points['pred_zs'] = latent_features
            end_points.update(end_points_meshes)


        end_points['parsed_predictions'] = parsed_predictions
        end_points['eval_dict'] = eval_dict
        return end_points

    def forward(self, data, export_shape=False):
        points = data['point_clouds']
        end_points = {}
        end_points = self.backbone(points, end_points)
        # -----QUAD DETECTION ------
        if self.phase == 'quad_detection' or self.phase == 'joint_detection':
            end_points = self.quad_detection(end_points)
        # --------- DETECTION ---------
        if self.phase == 'detection' or self.phase == 'joint_detection':
            end_points = self.detection(end_points)
        return end_points

    def loss(self, end_points, gt_data):
        if self.phase == 'quad_detection':
            total_loss = self.quad_detection_loss(end_points, gt_data,self.cfg.dataset_config)
            total_loss['total'] = total_loss['quad_total']
        elif self.phase == 'detection':
            total_loss = self.detection_loss(end_points, gt_data, self.cfg.dataset_config)
            total_loss['total'] = total_loss['object_total']

        elif self.phase == 'joint_detection':
            loss_sum = 0
            total_loss = {}
            loss_stages = ['detection', 'quad_detection']
            for loss_stage in loss_stages:
                if loss_stage == 'detection':
                    object_detection_loss = self.detection_loss(end_points, gt_data, self.cfg.dataset_config)
                    total_loss = {**total_loss, **object_detection_loss}
                    loss_sum += object_detection_loss['object_total']
                elif loss_stage == 'quad_detection':
                    quad_detection_loss = self.quad_detection_loss(end_points, gt_data,self.cfg.dataset_config)
                    total_loss = {**total_loss, **quad_detection_loss}
                    loss_sum += quad_detection_loss['quad_total']
            total_loss['total'] = loss_sum
        else:
            total_loss = {}
        return total_loss

    def generate_input(self, data):
        points = data['point_clouds']
        end_points = {}
        with torch.no_grad():
            end_points = self.backbone(points, end_points)
            end_points = self.detection(end_points)
            # end_points = self.quad_detection(end_points)
            anchors = end_points['anchors'][-1]
            anchor_sampled_pts_list,  original_pc_inds_list \
                    = self.SearchInstancePoints(data, anchors)
            end_points['anchor_sampled_pts_list'] = anchor_sampled_pts_list
            end_points['original_pc_inds_list'] = original_pc_inds_list
        return end_points


    def SearchInstancePoints(self, data, anchors):
        input_point_cloud = data['point_clouds'][:,:,:3]
        point_clouds_sampled_ids = data['choices'].detach().cpu().numpy()
        batch_size, nproposal, nanchor,_ = anchors.shape
        dist, _ = nn_distance_self(anchors.reshape(-1, nanchor, 3), square_root=True)  # BXK, 18
        dist, _ = dist.median(dim=-1)# BXK
        dist = dist.reshape(batch_size, nproposal)
        anchor_sampled_pts_list = []
        original_pc_inds_list = []
        for bid in range(batch_size):
            '''load data'''
            for nid in range(nproposal):
                # start = time()
                unique_sampled_point_indices, sampled_xyzs = QueryFromAnchors_per_proposal(
                    input_point_cloud[bid], anchors[bid][nid], dist[bid][nid], return_xyz=True, output_points_num = 500)
                unique_sampled_point_indices = unique_sampled_point_indices.detach().cpu().numpy()
                sampled_xyzs = sampled_xyzs.cpu().numpy()
                original_pc_inds = point_clouds_sampled_ids[bid][unique_sampled_point_indices]
                anchor_sampled_pts_list.append(sampled_xyzs.astype(np.float32))
                original_pc_inds_list.append(original_pc_inds)

        anchor_sampled_pts_list = np.array(anchor_sampled_pts_list)
        return anchor_sampled_pts_list,  original_pc_inds_list
