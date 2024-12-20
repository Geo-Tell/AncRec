# pointnet backbone
# author: ynie
# date: March, 2020
# cite: VoteNet
import torch
import torch.nn as nn
from models.registers import MODULES
import torch.nn.functional as F


@MODULES.register_module
class VotingModule(nn.Module):
    def __init__(self, cfg, optim_spec = None):
        '''
        Skeleton Extraction Net to obtain partial skeleton from a partial scan (refer to PointNet++).
        :param config: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(VotingModule, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Modules'''
        self.vote_factor = cfg.config['data']['vote_factor']
        self.in_dim = 256
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)

    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed * self.vote_factor
        net = F.relu(self.bn1(self.conv1(seed_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, (3+out_dim)*vote_factor, num_seed)

        net = net.transpose(2, 1).view(batch_size, num_seed, self.vote_factor, 3 + self.out_dim)
        offset = net[:, :, :, 0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)

        residual_features = net[:, :, :, 3:]  # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2, 1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2, 1).contiguous()

        return vote_xyz, vote_features


class PointGen(nn.Module):
    def __init__(self, bottleneck_size=128, output_dim = 3, scale=1.0):
        super(PointGen, self).__init__()

        self.scale = scale

        self.conv1 = torch.nn.Conv1d(bottleneck_size + 3, bottleneck_size//2, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)

        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//4)

        self.th = nn.Tanh()
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.scale*self.th(self.conv3(x))
        return x


class ShapeGen(nn.Module):
    def __init__(self, in_dims=128, subnetworks=1, scale=1.0, residual=True):
        super(ShapeGen, self).__init__()

        self.in_dims = in_dims
        self.subnetworks = subnetworks
        self.residual = residual

        self.decoders = nn.ModuleList([PointGen(bottleneck_size=4*in_dims, scale=scale)
                                       for i in range(0, self.subnetworks)])
        self.feat_enc = nn.Sequential(
                                    nn.Conv1d(in_dims, in_dims*2, 1),
                                    nn.BatchNorm1d(in_dims*2),
                                    nn.ReLU(True),
                                    nn.Conv1d(in_dims*2, in_dims*4, 1),
                                    nn.BatchNorm1d(in_dims*4),
                                    nn.ReLU(True)
                                      )

    def forward(self, xyz, features, vote_center, bs, proposal):
        # xyz (sphere) :  bs * self.num_proposal, 3, self.temp_pc_num2;
        # features: bs * self.num_proposal,cdim, 18;
        # vote_center: bs * self.num_proposal , 3
        out_shape_points = []
        current_shape_grid = xyz
        features = self.feat_enc(features)
        
        for i in range(self.subnetworks):
            current_feat = torch.cat([current_shape_grid, features], dim=1)
            if self.residual:
                current_shape_grid = current_shape_grid + self.decoders[i](current_feat)
            else:
                current_shape_grid = self.decoders[i](current_feat)

            # save deformed point cloud
            shape_points = current_shape_grid + vote_center
            out_shape_points.append(shape_points.reshape(bs, proposal, 3, -1).transpose(2,3).contiguous())
        return out_shape_points
            

# class ShapeGenVote(nn.Module):
#     def __init__(self, in_dims=128, subnetworks=1, scale=1.0, residual=True):
#         super(ShapeGenVote, self).__init__()

#         self.in_dims = in_dims
#         self.subnetworks = subnetworks
#         self.residual = residual

#         self.decoders = nn.ModuleList([PointGen(bottleneck_size=512, scale=scale)
#                                        for i in range(0, self.subnetworks)])
#         self.feat_enc = nn.Sequential(
#                                     nn.Conv1d(in_dims, 256, 1),
#                                     nn.BatchNorm1d(256),
#                                     nn.ReLU(True),
#                                     nn.Conv1d(256, 512, 1),
#                                     nn.BatchNorm1d(512),
#                                     nn.ReLU(True)
#                                       )

#     def forward(self, xyz, features, vote_center, bs, proposal):
#         out_shape_points = []
#         current_shape_grid = xyz - vote_center.transpose(1,2).contiguous()
#         features = self.feat_enc(features)
        
#         for i in range(self.subnetworks):
#             current_feat = torch.cat([current_shape_grid, features], dim=1)
#             if self.residual:
#                 current_shape_grid = current_shape_grid + self.decoders[i](current_feat)
#             else:
#                 current_shape_grid = self.decoders[i](current_feat)

#             # save deformed point cloud
#             shape_points = current_shape_grid.transpose(1,2).contiguous() + vote_center
#             out_shape_points.append(shape_points.reshape(bs, proposal, 3, -1))
#         return out_shape_points
class ShapeGen2(nn.Module):
    def __init__(self, in_dims=128, subnetworks=1, scale=1.0, residual=True):
        super(ShapeGen2, self).__init__()

        self.in_dims = in_dims
        self.subnetworks = subnetworks
        self.residual = residual

        self.decoders = nn.ModuleList([PointGen(bottleneck_size=4 * in_dims, scale=scale)
                                       for i in range(0, self.subnetworks)])
        self.feat_enc = nn.Sequential(
            nn.Conv1d(in_dims, in_dims * 2, 1),
            nn.BatchNorm1d(in_dims * 2),
            nn.ReLU(True),
            nn.Conv1d(in_dims * 2, in_dims * 4, 1),
            nn.BatchNorm1d(in_dims * 4),
            nn.ReLU(True)
        )

    def forward(self, pc_xyz, shape_feat, vote_feat):
        '''
        pc_xyz: B, nanchor, ngroup, 3
        shape_feat: B, nanchor, C
        '''
        out_shape_points = []
        current_shape_grid = xyz
        features = self.feat_enc(features)

        for i in range(self.subnetworks):
            current_feat = torch.cat([current_shape_grid, features], dim=1)
            if self.residual:
                current_shape_grid = current_shape_grid + self.decoders[i](current_feat)
            else:
                current_shape_grid = self.decoders[i](current_feat)

            # save deformed point cloud
            shape_points = current_shape_grid + vote_center
            out_shape_points.append(shape_points.reshape(bs, proposal, 3, -1).transpose(2, 3).contiguous())
        return out_shape_points