import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.registers import MODULES
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from external.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from models.anchorrec.modules.vote_module import ShapeGen
# from models.anchorrec.modules.CGNL import SpatialCGNL

SPHERE = np.load('spheres/sphere18.npy') # for detection and reconstruction

def decode_scores(net, end_points, num_heading_bin, num_size_cluster):
    net_transposed = net.transpose(2, 1)  # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:, :, 0:2]
    end_points['objectness_scores'] = objectness_scores

    base_xyz = end_points['aggregated_vote_xyz']  # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:, :, 2:5]  # (batch_size, num_proposal, 3)
    end_points['center'] = center

    heading_scores = net_transposed[:, :, 5:5 + num_heading_bin]
    heading_residuals_normalized = net_transposed[:, :, 5 + num_heading_bin:5 + num_heading_bin * 2]
    end_points['heading_scores'] = heading_scores  # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized
    # B x num_proposal x num_heading_bin (should be -1 to 1)

    size_scores = net_transposed[:, :, 5 + num_heading_bin * 2:5 + num_heading_bin * 2 + num_size_cluster]
    size_residuals_normalized = net_transposed[:, :,
                                5 + num_heading_bin * 2 + num_size_cluster:5 + num_heading_bin * 2 + num_size_cluster * 4].view(
        [batch_size, num_proposal, num_size_cluster, 3])  # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized

    sem_cls_scores = net_transposed[:, :, 5 + num_heading_bin * 2 + num_size_cluster * 4:]  # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores #5+24+32=
    return end_points


@MODULES.register_module
class ProposalModule(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        '''
        Skeleton Extraction Net to obtain partial skeleton from a partial scan (refer to PointNet++).
        :param config: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(ProposalModule, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.cfg = cfg
        '''Parameters'''
        self.num_class = cfg.dataset_config.num_class
        self.num_heading_bin = cfg.dataset_config.num_heading_bin
        self.num_size_cluster = cfg.dataset_config.num_size_cluster
        self.mean_size_arr = cfg.dataset_config.mean_size_arr
        self.num_proposal = cfg.config['data']['num_target']
        self.num_sample = cfg.config['model']['detection']['num_sample']
        self.sampling = cfg.config['model']['detection']['cluster_sampling']
        self.seed_feat_dim = 256
        self.hidden_dim = 128

        '''Modules'''
        # backbone branch
        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.8,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=256,
            radius=1.2,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])

        # voting
        self.vote_factor = cfg.config['data']['vote_factor']
        self.voting = nn.Sequential(
            nn.Conv1d(self.seed_feat_dim, self.seed_feat_dim, 1),
            nn.BatchNorm1d(self.seed_feat_dim),
            nn.ReLU(True),
            nn.Conv1d(self.seed_feat_dim, self.seed_feat_dim, 1),
            nn.BatchNorm1d(self.seed_feat_dim),
            nn.ReLU(True),
            nn.Conv1d(self.seed_feat_dim, (3 + self.seed_feat_dim) * self.vote_factor, 1)
        )

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=self.num_sample,
            mlp=[self.seed_feat_dim, 128, 128, self.hidden_dim],
            use_xyz=True,
            normalize_xyz=True,
            ret_grouped_xyz_pre=True
        )

        self.use_anchor = cfg.config['model']['detection']['shape']['supervise']
        if self.use_anchor:
            self.temp_pc_num = SPHERE.shape[1]
            self.sphere = torch.Tensor(SPHERE).float().unsqueeze(0).unsqueeze(0).cuda()

            self.shapegen = ShapeGen(self.hidden_dim,
                                     cfg.config['model']['detection']['shape']['subnetworks'],
                                     cfg.config['model']['detection']['shape']['scale'],
                                     cfg.config['model']['detection']['shape']['residual'])
            self.sampler = PointnetFPModule([256, 128, self.hidden_dim])
            self.aux_head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Conv1d(128, 2 + 3 + self.num_heading_bin * 2 + \
                          self.num_size_cluster * 4 + self.num_class, 1),
            )
            self.w1 = nn.Parameter(torch.ones((1, 2 + 3 + self.num_heading_bin * 2 + \
                                               self.num_size_cluster * 4 + self.num_class, 1)))
            self.w2 = nn.Parameter(torch.ones((1, 2 + 3 + self.num_heading_bin * 2 + \
                                               self.num_size_cluster * 4 + self.num_class, 1)))

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.predict_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 2 + 3 + self.num_heading_bin * 2 \
                      + self.num_size_cluster * 4 + self.num_class, 1) #128,2+3+self.num_heading_bin*2+self.num_size_cluster*4+self.num_class,1
        )


    def forward(self, end_points):

        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        seed_xyz = end_points['sa2_xyz']
        feat_dim = end_points['sa2_features'].shape[1] // 2
        sa2_features = end_points['sa2_features'][:,:feat_dim,:].contiguous()
        bs = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]

        xyz, features, fps_inds, _, _  = self.sa3(seed_xyz, sa2_features)  # this fps_inds is just 0,1,...,511
        # xyz, features, _, fps_inds, _, _  = self.sa3(seed_xyz, sa2_features)  # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds, _, _  = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        # xyz, features, _, fps_inds, _, _  = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'],
                            end_points['sa4_features'])
        seed_features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], sa2_features, features)
        end_points['seed_features'] = seed_features

        # ----------------voting ---------------------------
        num_vote = num_seed * self.vote_factor
        net = self.voting(seed_features)
        net = net.transpose(2, 1).view(bs, num_seed, self.vote_factor, 3 + self.seed_feat_dim)
        offset = net[:, :, :, 0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(bs, num_vote, 3)

        residual_features = net[:, :, :, 3:]  # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2, 1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(bs, num_vote, self.seed_feat_dim)
        vote_features = vote_features.transpose(2, 1).contiguous()  # (batch_size, self.seed_feat_dim, num_vote)
        features_norm = torch.norm(vote_features, p=2, dim=1)
        vote_features = vote_features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = vote_xyz

        #new_xyz, new_features, grouped_idx, inds, grouped_xyz_pre, grouped_features

        # -----------------proposal--------------------------
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds, grouped_xyz_pre, grouped_feat = self.vote_aggregation(vote_xyz, vote_features)
            # xyz, features, grouped_idx, fps_inds, grouped_xyz_pre, grouped_feat = self.vote_aggregation(vote_xyz, vote_features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps':
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _, grouped_xyz_pre, grouped_feat = self.vote_aggregation(vote_xyz, vote_features,
            # xyz, features, grouped_idx, _, grouped_xyz_pre, grouped_feat = self.vote_aggregation(vote_xyz, vote_features,
                                                                                    sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features,  _, grouped_xyz_pre, grouped_feat = self.vote_aggregation(vote_xyz, vote_features,
            # xyz, features, grouped_idx, _, grouped_xyz_pre, grouped_feat = self.vote_aggregation(vote_xyz, vote_features,
                                                                                    sample_inds)
        else:
            self.cfg.log_string('Unknown sampling strategy: %s. Exiting!' % (self.sampling))
            exit()
        end_points['vote_cluser_xyz'] = grouped_xyz_pre #1,3,256,16
        end_points['aggregated_vote_xyz'] = xyz  # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds  # (batch_size, num_proposal,)
        # end_points['vote_cluster_inds'] = grouped_idx  # (batch_size, num_proposal,)

        # --------- PROPOSAL GENERATION ---------
        net = self.predict_head(features)
        # end_points['proposal_features'] = net
        end_points['vote_features'] = features

        # --------- SURFACE POINTS ----------
        if self.use_anchor:
            vote_center = xyz.reshape(-1, 3).unsqueeze(-1).contiguous()
            agg_vote_feat = features.transpose(1, 2) \
                .reshape(bs * self.num_proposal, -1) \
                .unsqueeze(-1).repeat(1, 1, self.temp_pc_num).contiguous()

            sphere = self.sphere.repeat(bs, self.num_proposal, 1, 1).to(features.device)
            anchors = self.shapegen(sphere.reshape(bs * self.num_proposal, 3, self.temp_pc_num).contiguous(),
                                         agg_vote_feat,
                                         vote_center,
                                         bs=bs,
                                         proposal=self.num_proposal,
                                         )
            end_points['anchors'] = anchors

            shape_vote_feat = self.sampler(unknown=anchors[-1].reshape(bs, -1, 3).contiguous(),
                                           known=end_points['seed_xyz'],
                                           unknow_feats=None,
                                           known_feats=end_points['seed_features'],
                                           )  # (B, C, nsample)

            shape_feat = shape_vote_feat.reshape(bs, -1, self.num_proposal, self.temp_pc_num).contiguous()
            end_points['shape_vote_feat'] = shape_feat
            shape_feat = shape_feat.mean(-1)
            net_aux = self.aux_head(shape_feat)
            net = self.w1 * net + self.w2 * net_aux
            end_points['anchor_features'] = shape_feat
            # end_points['proposal_features'] = torch.cat([features, shape_feat], dim=1)

        end_points = decode_scores(net, end_points, self.num_heading_bin, self.num_size_cluster)


        # end_points['proposal_features'] = net
        return end_points

