## use different sa in quad proposal module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.anchorrec.modules.vote_module import ShapeGen, VotingModule
from models.registers import MODULES
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import  PointnetSAModuleVotes, PointnetFPModule, build_shared_mlp1d
from external.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from models.anchorrec.modules.CGNL import SpatialCGNL
SPHERE = np.load('spheres/sphere18.npy')
def decode_scores(net, end_points,quad_scores_head,center_head,normal_vector_head,size_head ):
    base_xyz = end_points['aggregated_vote_quad_xyz']
    quad_scores = quad_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 2)
    center = center_head(net).transpose(2, 1) + base_xyz  # (batch_size, num_proposal, 3)
    normal_vector = normal_vector_head(net).transpose(2, 1)
    normal_vector_norm = torch.norm(normal_vector, p=2)
    normal_vector = normal_vector.div(normal_vector_norm)
    size = size_head(net).transpose(2, 1)
    # direction = self.direction_head(net).transpose(2, 1)
    end_points['quad_scores'] = quad_scores
    end_points['quad_center'] = center  # (batch_size, num_proposal, 3)
    end_points['normal_vector'] = normal_vector
    end_points['quad_size'] = size
    return end_points

@MODULES.register_module
class QuadProposalModule(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        '''
        Skeleton Extraction Net to obtain partial skeleton from a partial scan (refer to PointNet++).
        :param cfg: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(QuadProposalModule, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.cfg = cfg
        self.num_proposal = cfg.config['data']['num_quad_proposal']
        '''vote aggregation'''
        self.num_sample = cfg.config['model']['quad_detection']['num_sample']
        self.sampling = cfg.config['data']['cluster_sampling']
        self.seed_feat_dim = 256
        self.hidden_dim = 128
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
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=self.num_sample,
            mlp=[self.seed_feat_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim],
            use_xyz=True,
            normalize_xyz=True,
            ret_grouped_xyz_pre=True
        )
        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])

        '''Modules'''
        # filter
        self.quad_filter = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            # nn.Conv1d(self.hidden_dim, 2 + 3 + self.num_heading_bin * 2 + \
            #           self.num_size_cluster * 4 + self.num_class, 1),
        )
        # Vote clustering
        self.temp_pc_num = SPHERE.shape[1]
        self.sphere = torch.Tensor(SPHERE).float().unsqueeze(0).unsqueeze(0)

        self.shapegen = ShapeGen(self.hidden_dim,
                                 cfg.config['model']['quad_detection']['shape']['subnetworks'],
                                 cfg.config['model']['quad_detection']['shape']['scale'],
                                 cfg.config['model']['quad_detection']['shape']['residual'])
        self.sampler = PointnetFPModule([256, self.hidden_dim, self.hidden_dim])



        self.vote = VotingModule(cfg)

        # self.fuse = build_shared_mlp1d([256, 128])

        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(self.hidden_dim, self.hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.hidden_dim, self.hidden_dim, 1)
        # self.conv3 = torch.nn.Conv1d(128, 2 + 3 + self.num_heading_bin * 2 + self.num_size_cluster * 4 + self.num_class,
        #                              1)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_dim)

        self.aux_head = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(True),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(True),
            # nn.Conv1d(self.hidden_dim, 2 + 3 + self.num_heading_bin * 2 + \
            #           self.num_size_cluster * 4 + self.num_class, 1),
        )
        self.w1 = nn.Parameter(torch.ones((1, self.hidden_dim, 1)))
        self.w2 = nn.Parameter(torch.ones((1, self.hidden_dim, 1)))
        self.quad_scores_head = torch.nn.Conv1d(self.hidden_dim, 2, 1)
        self.center_head = torch.nn.Conv1d(self.hidden_dim, 3, 1)
        self.normal_vector_head = torch.nn.Conv1d(self.hidden_dim, 3, 1)
        self.size_head = torch.nn.Conv1d(self.hidden_dim, 2, 1)
        if cfg.config['model']['quad_detection']['sa']:
            self.sa1 = SpatialCGNL(self.hidden_dim, int(self.hidden_dim/ 2), use_scale=False, groups=4)
            self.sa2 = SpatialCGNL(self.hidden_dim, int(self.hidden_dim / 2), use_scale=False, groups=4)
            self.gs_conv1 = torch.nn.Conv1d(512, self.hidden_dim, 1)

    def forward(self,end_points):

        """
        Args:
            seed_xyz: (B,K,3)
            seed_features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        sa2_xyz = end_points['sa2_xyz']
        feat_dim = end_points['sa2_features'].shape[1] // 2
        sa2_features = end_points['sa2_features'][:,feat_dim:,:].contiguous()
        xyz, features, fps_inds, _, _ = self.sa3(sa2_xyz, sa2_features)
        # xyz, features, _, fps_inds, _, _ = self.sa3(sa2_xyz, sa2_features)
        end_points['quad_sa3_xyz']= xyz
        end_points['quad_sa3_features'] = features
        xyz, features, fps_inds, _, _ = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        # xyz, features, _, fps_inds, _, _ = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points['quad_sa4_xyz'] = xyz
        end_points['quad_sa4_features'] = features
        features = self.fp1(end_points['quad_sa3_xyz'], end_points['quad_sa4_xyz'], end_points['quad_sa3_features'],
                            end_points['quad_sa4_features'])
        seed_features = self.fp2(end_points['sa2_xyz'], end_points['quad_sa3_xyz'], sa2_features, features)
        end_points['quad_seed_features'] = seed_features
        seed_xyz = end_points['sa2_xyz'] #256

        xyz, features = self.vote(seed_xyz, seed_features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_quad_xyz'] = xyz

        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds, grouped_xyz_pre, grouped_feat = self.vote_aggregation(xyz, features)
            # xyz, features, grouped_idx, fps_inds, grouped_xyz_pre, grouped_feat = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps':
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _, grouped_xyz_pre, grouped_feat = self.vote_aggregation(xyz, features, sample_inds)
            # xyz, features, _, _, grouped_xyz_pre, grouped_feat = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _, grouped_xyz_pre, grouped_feat = self.vote_aggregation(xyz, features, sample_inds)
            # xyz, features, _, _, grouped_xyz_pre, grouped_feat = self.vote_aggregation(xyz, features, sample_inds)
        else:
            self.cfg.log_string('Unknown sampling strategy: %s. Exiting!' % (self.sampling))
            exit()
        end_points['aggregated_vote_quad_xyz'] = xyz  # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_quad_inds'] = sample_inds  # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal
        # end_points['quad_vote_cluster_inds'] = grouped_idx  # (batch_size, num_proposal,)

        bs = xyz.shape[0]

        quad_center = xyz.reshape(-1, 3).unsqueeze(-1).contiguous()
        quad_feat = features.transpose(1, 2) \
            .reshape(bs * self.num_proposal, -1) \
            .unsqueeze(-1).repeat(1, 1, self.temp_pc_num).contiguous()

        # --------- SURFACE POINTS ----------
        sphere = self.sphere.repeat(bs, self.num_proposal, 1, 1).to(quad_feat.device)
        shape_points = self.shapegen(sphere.reshape(bs * self.num_proposal, 3, self.temp_pc_num).contiguous(),
                                     quad_feat,
                                     quad_center ,
                                     bs=bs,
                                     proposal=self.num_proposal
                                     )
        end_points['quad_surface_points'] = shape_points # list of shape points (bs,nproposal, 18, 3) from each subnetworks of shape points

        # the shape_points from the last subnetwork are used
        shape_vote_feat = self.sampler(unknown=shape_points[-1].reshape(bs, -1, 3).contiguous(),
                                       known= seed_xyz,
                                       unknow_feats=None,
                                       known_feats= seed_features
                                       )  # (B, C, nsample)

        shape_feat = shape_vote_feat.reshape(bs, -1, self.num_proposal, self.temp_pc_num).contiguous().mean(-1)

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net))) # 1024,128,18
        # net = self.conv3(net)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)
        net_aux = self.aux_head(shape_feat) # 8,128,256

        net = self.w1 * net + self.w2 * net_aux # bs, C, num_target

        if self.cfg.config['model']['quad_detection']['sa']:
            feature_dim = features.shape[1]
            features = features.contiguous().view(bs, feature_dim, 16, 16)  #B,128,Num_proposal(256)
            net = self.sa1(features)
            net = self.sa2(net)
            net = net.contiguous().view(bs, feature_dim, self.num_proposal)
            features = features.contiguous().view(bs, feature_dim, self.num_proposal)
            global_features_2 = F.max_pool1d(features, kernel_size=features.size(2))  # (B, 128, 1)
            global_features_1 = F.max_pool1d(seed_features, kernel_size=seed_features.size(2))  # (B, 256, 1) seed_feat_dim
            global_features = torch.cat((global_features_1, global_features_2), 1)  # (B, 256+128, 1) hidden_dim
            # global_features = torch.cat((global_features.expand(features.shape[0], 256 + 128, self.num_proposal), net), 1) # B, 256+128+128, 256
            global_features = torch.cat((global_features.repeat(1,1, self.num_proposal), net), 1) # B, 256+128+128, 256
            global_features = self.gs_conv1(global_features) # B,C,N
            global_features = torch.sigmoid(torch.log(torch.abs(global_features)+1e-8))
            net = net * global_features

        end_points = decode_scores(net, end_points, self.quad_scores_head, self.center_head, self.normal_vector_head, self.size_head )

        return end_points
