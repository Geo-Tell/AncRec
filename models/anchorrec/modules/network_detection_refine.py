import torch
from models.network import BaseNetwork
from models.registers import METHODS, MODULES, LOSSES
from net_utils.ap_helper import parse_predictions, parse_groundtruths, assembly_pred_map_cls, assembly_gt_map_cls
from models.loss import compute_anchor_loss, compute_objectness_loss, compute_objectness_loss_new, compute_box_and_sem_cls_loss
from time import time
import torch.nn as nn
NEAR_THRESHOLD = 0.3
from models.loss import soft_cross_entropy
criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]
# OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]
objectness_criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
@METHODS.register_module
class anchorrec_detection_refine(BaseNetwork):
    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg
        self.use_rgb = cfg.config.get('use_rgb',False)
        phase_names = ['img_encoder','detection_refine_head']
        '''load network blocks'''
        for phase_name, net_spec in cfg.config['model'].items():
            if phase_name not in phase_names:
                continue
            method_name = net_spec['method']
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            print(method_name)
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)

        '''freeze submodules or not'''
        self.freeze_modules(cfg)

    def forward(self, data):
        end_points = {}
        input_images = data['img']
        # valid_mask = data['valid_mask']
        input_sem_params = data['sem_params']
        # device = input_sem_params.device
        if len(input_images.shape) == 5:
            single_object_dataset = False
            bsize, nproposal, img_size, img_size, _ = input_images.shape
            input_images = input_images.view(-1, img_size, img_size,3).contiguous()
        else:
            single_object_dataset = True
            nproposal, img_size, img_size, _ = input_images.shape
            bsize = 1
        rgb_features = self.img_encoder(input_images) #B,C
        end_points['rgb_features'] = rgb_features
        #FusionPredictHead
        if type(self.detection_refine_head).__name__ == 'FusionPredictHead':
            objectness_scores, sem_cls_scores = self.detection_refine_head(input_sem_params.view(bsize*nproposal, -1).contiguous(), rgb_features)
        #FusionPredictHead_from_feature
        elif type(self.detection_refine_head).__name__ == 'FusionPredictHead_from_feature':
            vote_features = data['vote_features'].view(bsize * nproposal, -1).contiguous()
            anchor_features = data['anchor_features'].view(bsize * nproposal, -1).contiguous()
            objectness_scores, sem_cls_scores = self.detection_refine_head(vote_features,anchor_features,rgb_features)
        else:
            raise NotImplementedError
        if not single_object_dataset:
            end_points['objectness_scores'] = objectness_scores.view(bsize, nproposal, -1)
            end_points['sem_cls_scores'] = sem_cls_scores.view(bsize, nproposal, -1)
        else:
            end_points['objectness_scores'] = objectness_scores
            end_points['sem_cls_scores'] = sem_cls_scores
        return end_points

    def generate(self, data, baseline=False):
        end_points = {}
        # valid_mask = data['valid_mask']
        input_sem_params = data['sem_params']
        # device = input_sem_params.device
        if baseline:
            bsize, nproposal, _ = input_sem_params.shape
            end_points['objectness_scores'] = input_sem_params[:,:,:2]
            end_points['sem_cls_scores'] = input_sem_params[:,:,2:]
            # valid_mask = np.ones((bsize,nproposal)).astype(np.bool)
        else:
            input_images = data['img']
            # valid_mask = data['valid_mask'].astype(np.bool)
            bsize, nproposal, img_size, img_size, _ = input_images.shape
            input_images = input_images.view(-1, img_size, img_size,3).contiguous()
            with torch.no_grad():
                rgb_features = self.img_encoder(input_images) #, valid_mask)
                end_points['rgb_features'] = rgb_features.view(bsize, nproposal, -1)
                #FusionPredictHead
                if type(self.detection_refine_head).__name__ == 'FusionPredictHead':
                    objectness_scores, sem_cls_scores = self.detection_refine_head(
                        input_sem_params.view(bsize * nproposal, -1).contiguous(), rgb_features)
                # FusionPredictHead_from_feature
                elif type(self.detection_refine_head).__name__ == 'FusionPredictHead_from_feature':
                    vote_features = data['vote_features'].view(bsize * nproposal, -1).contiguous()
                    anchor_features = data['anchor_features'].view(bsize * nproposal, -1).contiguous()
                    objectness_scores, sem_cls_scores = self.detection_refine_head(vote_features, anchor_features, rgb_features)
                else:
                    raise NotImplementedError
                end_points['objectness_scores'] = objectness_scores.view(bsize, nproposal, -1)
                end_points['sem_cls_scores'] = sem_cls_scores.view(bsize, nproposal, -1)
        #for generate
        num_heading_bin = self.cfg.dataset_config.num_heading_bin
        num_size_cluster = self.cfg.dataset_config.num_size_cluster
        end_points = self.decode_scores_geo(data['geo_params'], data['aggregated_vote_xyz'], end_points,bsize, nproposal,num_heading_bin,num_size_cluster )
        eval_dict = {}
        parsed_predictions = {}
        parsed_predictions, eval_dict = parse_predictions(parsed_predictions, eval_dict, end_points, data,
                                                          self.cfg.eval_config)
        parsed_gts = parse_groundtruths(data, self.cfg.dataset_config)
        eval_dict['batch_gt_map_cls'] = assembly_gt_map_cls(parsed_gts)
        end_points['parsed_predictions'] = parsed_predictions
        end_points['eval_dict'] = eval_dict
        end_points['parsed_gts'] = parsed_gts
        return end_points

    @classmethod
    def decode_scores_geo(self, net, base_xyz, end_points, bsize, nproposal, num_heading_bin = 12, num_size_cluster = 8):
        center = base_xyz + net[:,:,:3] # (batch_size x num_proposal, 3)
        end_points['center'] = center

        heading_scores = net[:,:, 3:3 + num_heading_bin]
        heading_residuals_normalized = net[:, :, 3 + num_heading_bin:3 + num_heading_bin * 2]
        end_points['heading_scores'] = heading_scores  # Bxnum_proposalxnum_heading_bin
        end_points['heading_residuals_normalized'] = heading_residuals_normalized
        # B x num_proposal x num_heading_bin (should be -1 to 1)

        size_scores = net[:, :, 3 + num_heading_bin * 2:3 + num_heading_bin * 2 + num_size_cluster]
        size_residuals_normalized = net[:,:,
                                    3 + num_heading_bin * 2 + num_size_cluster:3 + num_heading_bin * 2 + num_size_cluster * 4].view(
            bsize, nproposal, num_size_cluster, 3)  # Bxnum_proposalxnum_size_clusterx3
        end_points['size_scores'] = size_scores
        end_points['size_residuals_normalized'] = size_residuals_normalized

        return end_points


    def loss(self, est_data, gt_data, use_recall_for_objectness = False, use_soft_loss=False):
        dataset_config = self.cfg.dataset_config
        loss_dict = {}
        valid_mask = gt_data['valid_mask']
        if 'objectness_label' in gt_data:
            sem_cls_label = gt_data['gt_sem_cls']
            objectness_label = gt_data['objectness_label']
            sem_cls_loss = criterion_sem_cls(est_data['sem_cls_scores'], sem_cls_label)  # (B,K)
            sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        else:
            # Obj loss
            est_data['aggregated_vote_xyz'] = gt_data['aggregated_vote_xyz']
            if use_recall_for_objectness:
                objectness_loss, objectness_label, object_assignment, soft_cls_labels, objectness_accuracy = \
                    compute_objectness_loss_new(est_data, gt_data, dataset_config, valid_mask=valid_mask)  # objectness_mask
            else:
                objectness_loss, objectness_label, object_assignment, objectness_recall, objectness_accuracy = \
                    compute_objectness_loss(est_data, gt_data, dataset_config, valid_mask=valid_mask)  # objectness_mask

            if use_soft_loss and use_recall_for_objectness:
                sem_cls_loss = soft_cross_entropy(est_data['sem_cls_scores'], soft_cls_labels, reduction=None)
                sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
            else:
                sem_cls_label = torch.gather(gt_data['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
                sem_cls_loss = criterion_sem_cls(est_data['sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
                sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        loss_dict['sem_cls_loss'] = sem_cls_loss.item()
        # loss_dict['objectness_recall'] = objectness_recall
        # loss_dict['objectness_accuracy'] = objectness_accuracy
        # loss_dict['objectness_loss'] = objectness_loss.item()

        loss = sem_cls_loss  #+  objectness_loss # +  0.5 *
        loss *= 10
        loss_dict['total'] = loss
        return loss_dict



