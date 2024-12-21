from torchvision.models import resnet50, resnet34
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt
from models.registers import MODULES
from net_utils.libs import softmax
class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level=0,
                 end_level=-1,
                 upsample_cfg=dict(mode='bilinear')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins

        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins

        self.start_level = start_level
        self.end_level = end_level


        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d( in_channels[i], out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels,3,padding=1)
            upsample_conv = nn.Conv2d(out_channels, out_channels,3,padding=1,stride=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.upsample_convs.append(upsample_conv)


    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]
        final_shape = laterals[0].shape[2:]
        final_outs = []
        for i in range(used_backbone_levels):
            out = self.fpn_convs[i](laterals[i])
            upsample_out = F.interpolate(out,size=final_shape, **self.upsample_cfg)
            final_outs.append(upsample_out)

        feats = torch.sum(torch.stack(final_outs, dim=0), dim=0)
        return feats

@MODULES.register_module
class ResNet_fpn(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, cfg, optim_spec=None):
        super(ResNet_fpn, self).__init__()
        self.optim_spec = optim_spec
        config = cfg.config['model']['img_encoder']
        out_channels = config['out_channels']
        frozen_stages = config['frozen_stages']
        use_point_mask = config['use_point_mask']
        self.use_point_mask = use_point_mask
        self.backbone = resnet34(pretrained=True)
        # self.backbone = resnet50(pretrained=True)
        self._freeze_stages(self.backbone,  frozen_stages)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.neck = FPN(in_channels=[1024, 2048], out_channels=out_channels) #in_channels=[256, 512, 1024, 2048]
        self.neck = nn.Sequential(
            nn.Linear(512,out_channels), #512 2048
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(out_channels, out_channels)
        ) #512: resnet34; 2048: resnet50

    def _freeze_stages(self, backbone, frozen_stages):
        if frozen_stages >= 0:
            backbone.bn1.eval()
            for m in [backbone.conv1, backbone.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, frozen_stages + 1):
            m = getattr(backbone, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def get_points_mask(self,w,h, points, sigma=10):
        batch_size = points.shape[0]
        mask = np.zeros((batch_size, w,h),np.uint8)
        points = points.cpu().numpy().astype(np.uint8)
        fov_labels = ((points[:, :, 0] <= h)
                      & (points[:, :, 0] > 0)
                      & (points[:, :, 1] <= w)
                      & (points[:, :, 1] > 0))
        for bid in range(batch_size):
            batch_pts = points[bid][fov_labels[bid]]
            mask[bid][batch_pts[:,0],batch_pts[:,1]] = 1

        max_dist = 255

        pos_points_mask_dist = distance_transform_edt(1 - mask)
        pos_points_mask_dist = torch.from_numpy(np.minimum(pos_points_mask_dist, max_dist))

        point_map = torch.exp(-2.772588722 * (pos_points_mask_dist ** 2) / (sigma ** 2)).to(torch.float32)
        return point_map #B,W,H,1

    def forward(self, rgb, pts2d=None):
        _, w, h, _ = rgb.shape
        if self.use_point_mask and pts2d != None:
            point_mask = self.get_points_mask(w,h,pts2d)
            point_mask = point_mask.to(rgb.device)
            rgb = rgb * point_mask.unsqueeze(-1)
        rgb = rgb.contiguous().permute(0, 3, 1, 2)
        x = nn.Sequential(*list(self.backbone.children())[:4])(rgb) #conv2d, bn2d, relu, maxpool
        out_feats = []
        for i in range(1,5):
            m = getattr(self.backbone, f'layer{i}')
            x = m(x)
            if i > 2:
                out_feats.append(x)

        # img_features = self.neck(out_feats) #B,C,n,n
        # img_features = self.avgpool(img_features).squeeze(-1).squeeze(-1)  #B,C,1,1
        img_features = self.avgpool(x).squeeze(-1).squeeze(-1)  #B,C,1,1
        img_features = self.neck(img_features)
        return img_features

@MODULES.register_module
class FusionPredictHead_no_rgb(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(FusionPredictHead_no_rgb, self).__init__()
        self.optim_spec = optim_spec
        config = cfg.config['model']['detection_head']
        proposal_input_cdim = config['proposal_feat_channels']
        self.use_anchor = config['use_anchor']
        hidden_dim = config['hidden_dim']
        self.num_class = cfg.dataset_config.num_class
        self.num_heading_bin = cfg.dataset_config.num_heading_bin
        self.num_size_cluster = cfg.dataset_config.num_size_cluster
        self.mean_size_arr = cfg.dataset_config.mean_size_arr

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # head layers
        self.semantic_head = nn.Sequential(
            nn.Linear(proposal_input_cdim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),  # //2
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 2 + self.num_class)  # objectness, size_cls, sem_cls
        )
        self.geometric_head = nn.Sequential(
            nn.Linear(proposal_input_cdim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 3 + self.num_heading_bin * 2 \
                      + self.num_size_cluster * 4)  # objectness, headings, size_reg
        )
        self.w_sem_vote = nn.Parameter(torch.ones((1, self.num_class + 2)))
        self.w_geo_vote = nn.Parameter(torch.ones((1, 3 + self.num_heading_bin * 2 + self.num_size_cluster * 4)))
        self.criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        self.criterion_objectness = nn.CrossEntropyLoss(reduction='none')

        if self.use_anchor:
             # head layers
            self.semantic_head_aux = nn.Sequential(
                nn.Linear(proposal_input_cdim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, 2 + self.num_class)  # objectness, size_cls, sem_cls
            )
            self.geometric_head_aux = nn.Sequential(
                nn.Linear(proposal_input_cdim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, 3 + self.num_heading_bin * 2 \
                          + self.num_size_cluster * 4)  # objectness, headings, size_reg
            )
            self.w_geo_anchor = nn.Parameter(torch.ones((1, 3 + self.num_heading_bin * 2 + self.num_size_cluster * 4)))
            self.w_sem_anchor = nn.Parameter(torch.ones((1, self.num_class + 2)))


    def forward(self, end_points, num_heading_bin, num_size_cluster):
        vote_features = end_points['vote_features']  # B,C,N
        bsize, feat_dim, nproposal = vote_features.shape
        vote_features = vote_features.transpose(1, 2).contiguous().reshape(-1, feat_dim)  # BxN, C
        # rgb_features = self.avgpool(end_points['rgb_features']).squeeze() # BxN, C'

        if 'anchor_features' in end_points and self.use_anchor:
            anchor_features = end_points['anchor_features']
            feat_dim = anchor_features.shape[1]
            anchor_features = anchor_features.transpose(1, 2).contiguous().reshape(-1, feat_dim)  # BxN, C
            geo_params = (self.w_geo_vote * self.geometric_head(vote_features)
                          + self.w_geo_anchor * self.geometric_head_aux(anchor_features)) \
                         / (self.w_geo_vote + self.w_geo_anchor)
            sem_params = (self.w_sem_vote * self.semantic_head(vote_features)
                          + self.w_sem_anchor * self.semantic_head_aux(anchor_features)) \
                 /(self.w_sem_vote + self.w_sem_anchor) #
        else:
            geo_params  = self.geometric_head(vote_features)
            sem_params =  self.semantic_head(vote_features)
        end_points['geo_params'] = geo_params
        end_points['sem_params'] = sem_params
        end_points = self.decode_scores_geo(geo_params, end_points, num_heading_bin, num_size_cluster, bsize, nproposal)
        end_points['objectness_scores'] = sem_params[:, :2].view(bsize, nproposal, -1)
        end_points['sem_cls_scores'] = sem_params[:, 2:].view(bsize, nproposal, -1)
        return end_points

    def decode_scores_geo(self, net, end_points, num_heading_bin, num_size_cluster, bsize, nproposal):
        base_xyz = end_points['aggregated_vote_xyz']  # (batch_size, num_proposal, 3)
        center = base_xyz + net[:, :3].view(bsize, nproposal, 3).contiguous()  # (batch_size x num_proposal, 3)
        end_points['center'] = center

        heading_scores = net[:, 3:3 + num_heading_bin]
        heading_residuals_normalized = net[:, 3 + num_heading_bin:3 + num_heading_bin * 2]
        end_points['heading_scores'] = heading_scores.view(bsize, nproposal, -1)  # Bxnum_proposalxnum_heading_bin
        end_points['heading_residuals_normalized'] = heading_residuals_normalized.view(bsize, nproposal, -1)
        # B x num_proposal x num_heading_bin (should be -1 to 1)

        size_scores = net[:, 3 + num_heading_bin * 2:3 + num_heading_bin * 2 + num_size_cluster].view(bsize, nproposal,
                                                                                                      -1)
        size_residuals_normalized = net[:,
                                    3 + num_heading_bin * 2 + num_size_cluster:3 + num_heading_bin * 2 + num_size_cluster * 4].view(
            bsize, nproposal, num_size_cluster, 3)  # Bxnum_proposalxnum_size_clusterx3
        end_points['size_scores'] = size_scores
        end_points['size_residuals_normalized'] = size_residuals_normalized
        return end_points

@MODULES.register_module
class FusionPredictHead(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(FusionPredictHead, self).__init__()
        self.optim_spec = optim_spec
        config = cfg.config['model']['detection_refine_head']
        hidden_dim  = config['hidden_dim']
        self.num_class = cfg.dataset_config.num_class
        self.num_heading_bin = cfg.dataset_config.num_heading_bin
        self.num_size_cluster = cfg.dataset_config.num_size_cluster
        self.mean_size_arr = cfg.dataset_config.mean_size_arr
        # self.w_sem_vote = nn.Parameter(torch.ones((1, self.num_class + 2)))
        self.w_sem_rgb = nn.Parameter(torch.ones((1,  self.num_class + 2)))
        # self.w_geo_vote = nn.Parameter(torch.ones((1, 3 + self.num_heading_bin * 2 + self.num_size_cluster * 4)))
        rgb_input_cdim = config['rgb_input_channels']
        self.rgb_predict_head = nn.Sequential(
            nn.Linear(rgb_input_cdim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, 2 + self.num_class)
        )
        self.criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        self.criterion_objectness  = nn.CrossEntropyLoss(reduction='none')

    def forward(self, sem_params, rgb_features):
        sem_params = (sem_params + self.w_sem_rgb * self.rgb_predict_head(rgb_features)) \
                     /  (self.w_sem_rgb +1)
        # sem_params =  self.rgb_predict_head(rgb_features)
        objectness_scores  = sem_params[:,:2]
        sem_cls_scores  = sem_params[:,2:]
        return objectness_scores, sem_cls_scores

    def loss(self, objectness_scores, sem_cls_scores, data):
        loss = {}
        objectness_label = data['objectness_label'][data['valid_mask'].bool()] #B
        sem_cls_labels = data['gt_sem_cls_labels'][data['valid_mask'].bool()]
        # soft_cls_labels = data['soft_cls_label'][data['valid_mask'].bool()]
        # point_recalls = data['point_recall'][data['valid_mask'].bool()]
        data_num = objectness_label.shape[0]

        sem_cls_loss = self.criterion_sem_cls(sem_cls_scores, sem_cls_labels)
        # sem_cls_loss = soft_cross_entropy(sem_cls_scores, soft_cls_labels, reduction=None)
        # sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        objectness_loss = self.criterion_objectness(objectness_scores, objectness_label.long())
        # soft_objectness_label = torch.cat([1-point_recalls .unsqueeze(1), point_recalls.unsqueeze(1)],1)
        # objectness_loss = soft_cross_entropy(objectness_scores, soft_objectness_label)
        # objectness_loss = torch.mean(objectness_loss)
        loss['sem_cls_loss'] = sem_cls_loss
        loss['objectness_loss'] = objectness_loss

        obj_logits = objectness_scores.detach().cpu().numpy()
        obj_prob = softmax(obj_logits)[:, 1]
        num_obj_correct = (objectness_label.cpu().numpy() == (obj_prob > 0.3)).sum()

        pred_sem_cls = torch.argmax(sem_cls_scores, -1)
        num_cls_correct = (pred_sem_cls[objectness_label] == sem_cls_labels[objectness_label]).sum()

        loss['objectness_accuracy'] = (num_obj_correct / data_num).item()
        loss['cls_accuracy'] = (num_cls_correct / objectness_label.sum()).item()


        total_loss = loss['sem_cls_loss'] + 5 * loss['objectness_loss']
        # for key in loss:
        #     if 'loss' in key:
        #         total_loss += loss[key]
        loss['total'] = total_loss
        return loss

@MODULES.register_module
class FusionPredictHead_from_feature(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(FusionPredictHead_from_feature, self).__init__()
        self.optim_spec = optim_spec
        config = cfg.config['model']['detection_refine_head']
        hidden_dim  = config['hidden_dim']
        rgb_input_cdim = config['rgb_input_channels']
        proposal_input_cdim = config['proposal_input_cdim']
        self.num_class = cfg.dataset_config.num_class
        self.num_heading_bin = cfg.dataset_config.num_heading_bin
        self.num_size_cluster = cfg.dataset_config.num_size_cluster
        self.mean_size_arr = cfg.dataset_config.mean_size_arr

        self.w_sem_vote = nn.Parameter(torch.ones((1, self.num_class + 2)))
        self.w_sem_anchor = nn.Parameter(torch.ones((1,  self.num_class + 2)))
        self.w_sem_rgb = nn.Parameter(torch.ones((1,  self.num_class + 2)))
        self.rgb_predict_head = nn.Sequential(
            nn.Linear(rgb_input_cdim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, 2 + self.num_class)
        )
        self.semantic_head_vote = nn.Sequential(
            nn.Linear(proposal_input_cdim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),  # //2
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 2 + self.num_class)  # objectness, size_cls, sem_cls
        )
        self.semantic_head_anchor = nn.Sequential(
            nn.Linear(proposal_input_cdim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),  # //2
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 2 + self.num_class)  # objectness, size_cls, sem_cls
        )

        # self.criterion_objectness =  nn.BCELoss()
        self.criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        self.criterion_objectness  = nn.CrossEntropyLoss(reduction='none')
        # self.sigmoid = nn.Sigmoid()

    def forward(self, vote_features, anchor_features, rgb_features):
        sem_params = (  self.w_sem_vote *   self.semantic_head_vote(vote_features)
                      + self.w_sem_anchor * self.semantic_head_anchor(anchor_features)
                      + self.w_sem_rgb *    self.rgb_predict_head(rgb_features)) \
                     / (self.w_sem_vote +   self.w_sem_anchor + self.w_sem_rgb)
        # sem_params =  self.rgb_predict_head(rgb_features)
        objectness_scores  = sem_params[:,:2]
        sem_cls_scores  = sem_params[:,2:]
        return objectness_scores, sem_cls_scores


