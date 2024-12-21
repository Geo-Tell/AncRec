import torch
from models.network import BaseNetwork
from models.registers import METHODS, MODULES, LOSSES
from models.anchorrec.SingleObjectDatasetWithRGB import MEAN_COLOR_RGB
from net_utils.ap_helper import parse_predictions, parse_groundtruths, assembly_pred_map_cls, assembly_gt_map_cls
from net_utils.quad_ap_helper import parse_quad_groundtruths, parse_quad_predictions
import numpy as np
import os
import cv2
from utils.scannet.extract_posed_images import adjust_image_range
@METHODS.register_module
class anchorrec(BaseNetwork):
    def __init__(self, cfg):
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        self.mode =  cfg.config['mode']
        self.phase = cfg.config[self.mode]['phase']
        self.test_quad = False
        phase_names = []
        print("phase name: %s"%self.phase)
        if self.phase == 'detection':
            phase_names += ['backbone', 'detection', 'detection_head']
        elif self.phase == 'quad_detection':
            self.test_quad = True
            phase_names += ['backbone', 'quad_detection']
        elif self.phase == 'joint_detection':
            self.test_quad = True
            phase_names += ['backbone', 'quad_detection', 'detection', 'detection_head']
        elif self.phase == 'prepare_data':
            phase_names += ['backbone', 'detection', 'detection_head', 'GenerateInstanceFeat']
        elif self.phase == "detection_with_rgb":
            phase_names += ['backbone', 'detection', 'detection_head', 'GenerateInstanceFeat', 'img_encoder','detection_refine_head']
            if cfg.config[self.mode].get('test_quad', False) == True:
                phase_names += ['quad_detection']
                self.test_quad = True
            else:
                self.test_quad = False
        elif self.phase == 'final_recon':
            phase_names += ['backbone', 'detection', 'detection_head', 'GenerateInstanceFeat','feature_encode','completion']
        elif self.phase == 'final_recon_with_rgb':
            phase_names += ['backbone', 'detection', 'detection_head','GenerateInstanceFeat','img_encoder', 'detection_refine_head', 'feature_encode','completion']

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
            if self.phase == 'quad_detection' or self.phase == 'joint_detection' or self.test_quad == True:
                end_points = self.quad_detection(end_points)
            # --------- DETECTION ---------
            if self.phase == 'detection' or self.phase == 'joint_detection':
                num_heading_bin = self.cfg.dataset_config.num_heading_bin
                num_size_cluster = self.cfg.dataset_config.num_size_cluster
                end_points = self.detection(end_points)
                end_points = self.detection_head(end_points, num_heading_bin, num_size_cluster)

        if self.phase == 'quad_detection' or self.phase == 'joint_detection' or self.test_quad == True:
            has_quad_ind = data['use_quad'].bool()
            parsed_predictions, eval_dict = parse_quad_predictions(parsed_predictions, eval_dict, end_points,
                                                                   has_quad_ind, self.cfg)
            eval_dict = parse_quad_groundtruths(eval_dict, data)

        if self.phase == 'detection' or self.phase == 'joint_detection':
            parsed_predictions, eval_dict = parse_predictions(parsed_predictions, eval_dict, end_points, data,
                                                              self.cfg.eval_config)

        elif self.phase == "final_recon":
            end_points = self.generate_input(data, test_set=True, use_rgb = False)
            parsed_predictions, eval_dict = parse_predictions(parsed_predictions, eval_dict, end_points, data,
                                                              self.cfg.eval_config, return_batch_pred_map_cls=False)
            pred_mask = eval_dict['pred_mask']
            if 'above_floor_mask' in parsed_predictions:
                pred_mask = pred_mask * parsed_predictions['above_floor_mask']
            obj_probs = parsed_predictions['obj_prob']
            dump_thresh = self.cfg.config[self.mode].get('dump_thresh', 0.05)
            sampled_ids = (obj_probs > dump_thresh) * pred_mask  # B,N
            end_points['sampled_ids'] = np.where(sampled_ids[0])[0]
            vote_features = end_points['vote_features']  # B,C
            anchor_features = end_points['anchor_features']  # B,C
            proposal_features = torch.cat([vote_features, anchor_features], 1)
            input_points = torch.tensor(end_points['anchor_sampled_pts_list']).to(vote_features.device)  # B,Npt,3

            if type(self.feature_encode).__name__ == 'ResnetPointnet':
                input_features = torch.cat(  # B,Npts, C+3
                    [input_points, proposal_features.unsqueeze(1).repeat(1, input_points.shape[1], 1)], dim=2)
                latent_features = self.recon(input_features)
            elif type(self.feature_encode).__name__ == 'PCTransformer_small':
                latent_features = self.feature_encode(input_points, proposal_features[0].t())
            sampled_ids = torch.tensor(sampled_ids[0]).to(vote_features.device)
            end_points_meshes = self.completion.generate(latent_features[sampled_ids.bool])
            end_points['pred_zs'] = latent_features
            end_points.update(end_points_meshes)

        elif self.phase == "final_recon_with_rgb" or self.phase == "detection_with_rgb":
            end_points = self.generate_input(data, test_set=True, use_rgb = True)
            device = end_points["anchors"][-1].device
            bsize, nproposal = end_points["anchors"][-1].shape[0], end_points["anchors"][-1].shape[1]
            assert bsize ==1
            bid = 0
            output_image_paths = end_points['output_image_paths']
            output_pts2d_range_list = end_points['output_pts2d_range_list']
            valid_mask = end_points['valid_mask'] #No image / pts2d_range recorded where valid_mask =False
            output_image_paths_all = []
            output_pts2d_range_list_all = []
            iii = 0
            for pth_id in range(nproposal):
                if valid_mask[pth_id]:
                    output_image_paths_all.append(output_image_paths[iii])
                    output_pts2d_range_list_all.append(output_pts2d_range_list[iii])
                    iii += 0
                else:
                    output_image_paths_all.append([])
                    output_pts2d_range_list_all.append([])

            min_pts2d_range = self.cfg.config['data'].get('min_pts2d_range', 20)
            image_size = self.cfg.config['data'].get('image_size', 400)
            new_valid_mask = np.ones(nproposal)
            image_list = []
            for nid in range(nproposal):
                if not valid_mask[nid]:
                    new_valid_mask[nid]=0
                    continue
                pts2d_range = output_pts2d_range_list_all[nid]
                minx, miny, maxx, maxy = pts2d_range
                if (maxx - minx) <  min_pts2d_range or (maxy - miny) <  min_pts2d_range:
                    new_valid_mask[nid] = 0
                    continue
                if not os.path.exists(output_image_paths_all[nid]):
                    new_valid_mask[nid] = 0
                    continue
                image = cv2.imread(output_image_paths_all[nid])
                if image is None:
                    continue
                new_range = adjust_image_range(image, pts2d_range, margin=10)
                minx, miny, maxx, maxy = new_range
                cropped_image = image[miny: maxy, minx: maxx]  # + 1
                cropped_image = cv2.resize(cropped_image, ( image_size,  image_size))
                cropped_image = (cropped_image - MEAN_COLOR_RGB) / 255.0
                cropped_image = torch.tensor(cropped_image.astype(np.float32)).unsqueeze(0)
                image_list.append(cropped_image)

            batch_images = torch.cat(image_list).to(device)
            new_valid_mask = new_valid_mask.astype(np.bool_)
            vote_features = end_points['vote_features'][bid].t()[new_valid_mask] # N,C
            anchor_features = end_points['anchor_features'][bid].t()[new_valid_mask]# N,C
            input_points = torch.tensor(end_points['anchor_sampled_pts_list'])[new_valid_mask].to(device)
            rgb_features = self.img_encoder(batch_images)
            '''predict coarse proposals'''
            new_nproposal = new_valid_mask.sum()
            geo_params = end_points['geo_params'][new_valid_mask] #N, L
            end_points['aggregated_vote_xyz'] =  end_points['aggregated_vote_xyz'][bid][new_valid_mask].unsqueeze(0)
            end_points = self.detection_head.decode_scores_geo(geo_params, end_points, bsize=1, nproposal=new_nproposal,
                                                             num_heading_bin=self.cfg.dataset_config.num_heading_bin,
                                                             num_size_cluster=self.cfg.dataset_config.num_size_cluster)
            sem_params = end_points['sem_params'][new_valid_mask] #N, L

            '''refine proposals by RGB features'''
            objectness_scores, sem_cls_scores = self.detection_refine_head(sem_params, rgb_features) ##N, L
            end_points['objectness_scores'] = objectness_scores.view(bsize, new_nproposal, -1)
            end_points['sem_cls_scores'] = sem_cls_scores.view(bsize,new_nproposal, -1)

            '''parse proposals into bounding boxes'''
            parsed_predictions, eval_dict = parse_predictions(parsed_predictions, eval_dict, end_points, data,
                                                              self.cfg.eval_config, return_batch_pred_map_cls= (self.phase != "final_recon_with_rgb"))

            if self.phase == "final_recon_with_rgb":
                '''predict shape feautre for reconstruction'''
                pred_mask = eval_dict['pred_mask']
                if 'above_floor_mask' in parsed_predictions:
                    pred_mask = pred_mask * parsed_predictions['above_floor_mask']
                obj_probs = parsed_predictions['obj_prob']
                dump_thresh = self.cfg.config[self.mode].get('dump_thresh', 0.05)
                final_mask = (obj_probs > dump_thresh) * pred_mask  # B,N
                end_points['sampled_ids'] = np.where(final_mask[bid])[0]
                final_mask = final_mask[bid]
                proposal_features = torch.cat([vote_features, anchor_features, rgb_features], 1) #N,C
                proposal_features = proposal_features[final_mask] #N,C
                input_points = input_points[final_mask]
                if type(self.feature_encode).__name__ == 'ResnetPointnet':
                    input_features = torch.cat(  # B,Npts, C+3
                        [input_points, proposal_features.unsqueeze(1).repeat(1, input_points.shape[1], 1)], dim=2)
                    latent_features = self.feature_encode(input_features)
                elif type(self.feature_encode).__name__ == 'PCTransformer_small':
                    latent_features = self.feature_encode(input_points, proposal_features)
                end_points_meshes = self.completion.generate(latent_features)
                end_points['pred_zs'] = latent_features
                end_points.update(end_points_meshes)

        if self.phase != "final_recon_with_rgb":
            parsed_gts = parse_groundtruths(data, self.cfg.dataset_config)
            eval_dict['batch_gt_map_cls'] = assembly_gt_map_cls(parsed_gts)
            end_points['parsed_gts'] = parsed_gts
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
            num_heading_bin = self.cfg.dataset_config.num_heading_bin
            num_size_cluster = self.cfg.dataset_config.num_size_cluster
            end_points = self.detection(end_points)
            end_points = self.detection_head(end_points, num_heading_bin, num_size_cluster)

        '''We train the RGB and reconstruction branch in network_detection_refine.py and network_small.py'''
        return end_points

    def loss(self, end_points, gt_data):
        if self.phase == 'quad_detection':
            total_loss = self.quad_detection_loss(end_points, gt_data,self.cfg.dataset_config)
            total_loss['total'] = total_loss['quad_total']
        elif self.phase == 'detection':
            total_loss = self.detection_head_loss(end_points, gt_data, self.cfg.dataset_config)
            total_loss['total'] = total_loss['object_total']

        elif self.phase == 'joint_detection':
            loss_sum = 0
            total_loss = {}
            loss_stages = ['detection', 'quad_detection']
            for loss_stage in loss_stages:
                if loss_stage == 'detection':
                    object_detection_loss = self.detection_head_loss(end_points, gt_data, self.cfg.dataset_config)
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

    def generate_input(self, data, test_set = False, use_rgb = True):
        points = data['point_clouds']
        end_points = {}
        num_heading_bin = self.cfg.dataset_config.num_heading_bin
        num_size_cluster = self.cfg.dataset_config.num_size_cluster
        with torch.no_grad():
            end_points = self.backbone(points, end_points)
            end_points = self.detection(end_points)
            end_points = self.detection_head(end_points, num_heading_bin, num_size_cluster)
            if self.test_quad:
                end_points = self.quad_detection(end_points)
            anchors = end_points['anchors'][-1]

            if use_rgb:
                if test_set:
                    anchor_sampled_pts_list, output_image_paths, output_pts2d_range_list, valid_mask, original_pc_inds_list \
                    = self.GenerateInstanceFeat.generate(data, anchors)
                else:
                    anchor_sampled_pts_list, output_image_paths, output_pts2d_range_list, valid_mask, original_pc_inds_list\
                   = self.GenerateInstanceFeat(data,anchors ,augment = False)
                end_points['output_image_paths'] = output_image_paths
                end_points['output_pts2d_range_list'] = output_pts2d_range_list
                end_points['valid_mask'] = valid_mask
            else:
                anchor_sampled_pts_list,  original_pc_inds_list \
                    = self.GenerateInstanceFeat(data, anchors, augment=False, return_images =False)
            end_points['anchor_sampled_pts_list'] = anchor_sampled_pts_list
            end_points['original_pc_inds_list'] = original_pc_inds_list

        return end_points
