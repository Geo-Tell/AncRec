from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from external.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
# from models.iscnet.modules.vote_module import ShapeGenDenser

def build_shared_mlp1d(mlp_spec: List[int], bn: bool = True, last_act: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv1d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm1d(mlp_spec[i]))
        if not last_act and i == len(mlp_spec)-1:
            continue
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)

def build_shared_mlp(mlp_spec: List[int], bn: bool = True, last_act: bool=True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        if not last_act and i == len(mlp_spec) - 1:
            continue
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True, sample_uniformly=False):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetSAModuleVotes(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None,  # for RBF pooling
            normalize_xyz: bool = False,  # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False,
            ret_grouped_xyz_pre: bool = False,
            ret_grouped_idx: bool = False
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.ret_grouped_xyz_pre = ret_grouped_xyz_pre
        self.ret_grouped_idx = ret_grouped_idx

        if npoint is not None:
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
                                                         use_xyz=use_xyz,
                                                         normalize_xyz=normalize_xyz,
                                                         sample_uniformly=sample_uniformly
                                                        )
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = build_shared_mlp(mlp_spec, bn=bn)

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint) if self.npoint is not None else None
        else:
            assert (inds.shape[1] == self.npoint)
        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped, inds
        ).transpose(1, 2).contiguous() if self.npoint is not None else None

        groupe_results = self.grouper(xyz, new_xyz, features)
        grouped_features = groupe_results['new_features']
        grouped_xyz = groupe_results['grouped_xyz']
        grouped_xyz_pre = groupe_results['grouped_xyz_pre']
        grouped_idx = groupe_results['idx']


        new_features = self.mlp_module(
            grouped_features
        )  # (B, mlp[-1], npoint, nsample)
        if self.pooling == 'max':
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'avg':
            new_features = F.avg_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'rbf':
            # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
            # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            rbf = torch.exp(
                -1 * grouped_xyz.pow(2).sum(1, keepdim=False) / (self.sigma ** 2) / 2)  # (B, npoint, nsample)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(
                self.nsample)  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        if self.ret_unique_cnt:
            return new_xyz, new_features, grouped_idx, inds, groupe_results['unique_cnt'] , grouped_xyz_pre, grouped_features
        elif self.ret_grouped_idx:
            return new_xyz, new_features, grouped_idx, inds, grouped_xyz_pre, grouped_features
        else:
            return new_xyz, new_features, inds, grouped_xyz_pre, grouped_features



class PointNextSAModuleVotes(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp_spec: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            is_head: bool = False,
            use_xyz: bool = True,
            normalize_xyz: bool = False,  # noramlize local XYZ with radius
            use_res: bool = True
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.is_head = is_head
        self.use_xyz = use_xyz
        self.normalize_xyz = normalize_xyz
        self.use_res = use_res
        self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
                                                     use_xyz=use_xyz, ret_grouped_xyz=True,
                                                     normalize_xyz=normalize_xyz
                                                         )

        if is_head:
            self.mlp_module = build_shared_mlp1d(mlp_spec, last_act=False)
        else:
            self.mlp_module = build_shared_mlp(mlp_spec, last_act= self.use_res)
        if self.use_res:
            if mlp_spec[0] == mlp_spec[-1]:
                self.skipconv = nn.Identity()
            else:
                channels = [mlp_spec[0],mlp_spec[-1]]
                self.skipconv = build_shared_mlp1d(channels, last_act = False)
            self.act = nn.ReLU()

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """
        xyz_flipped = xyz.transpose(1, 2).contiguous() #B,3,N
        # features= torch.cat(xyz_flipped, features)
        if self.is_head:
            new_features = self.mlp_module(features)
            inds = None

        else:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint) if self.npoint is not None else None
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped, inds
            ).transpose(1, 2).contiguous() if self.npoint is not None else None
            grouped_features, grouped_xyz = self.grouper(
                xyz, new_xyz, features
            )

            if self.use_res:
                identity = torch.gather(
                    features, -1, inds.long().unsqueeze(1).expand(-1, features.shape[1], -1))
                identity = self.skipconv(identity)

            new_features = self.mlp_module(grouped_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            if self.use_res:
                new_features = self.act(new_features + identity)
            xyz = new_xyz

        return xyz, new_features, inds

class InvResMLP(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            in_channels,
            expansion: int = 1,
            num_posconvs: int = 2,
            radius: float = None,
            nsample: int = None,
            use_xyz: bool = True,
            use_res: bool = True,
            normalize_xyz: bool = False,  # noramlize local XYZ with radius
    ):
        super().__init__()

        self.radius = radius
        self.nsample = nsample
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.use_res = use_res
        self.normalize_xyz = normalize_xyz
        mid_channels = in_channels * expansion

        if num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]

        self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
                                                         use_xyz=use_xyz, ret_grouped_xyz=True,
                                                         normalize_xyz=normalize_xyz)


        self.mlp_module = build_shared_mlp([in_channels, in_channels])
        self.pwconv = build_shared_mlp1d(channels, last_act = False)
        self.act = nn.ReLU()

    def forward(self, xyz: torch.Tensor,
                feat: torch.Tensor = None,
                ) -> (torch.Tensor, torch.Tensor):

        identity = feat # BCN
        grouped_feat, grouped_xyz = self.grouper(
            xyz, xyz, feat #B,N,3
        )

        new_features = self.mlp_module(grouped_feat)  # (B, mlp[-1], npoint, nsample)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)

        feat = self.pwconv(new_features.squeeze(-1)) #BCN -> BCN
        if self.use_res:
            feat = self.act(feat +identity)
        return feat



class PointnetSAModuleMSGVotes(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlps: List[List[int]],
            npoint: int,
            radii: List[float],
            nsamples: List[int],
            bn: bool = True,
            use_xyz: bool = True,
            sample_uniformly: bool = False
    ):
        super().__init__()

        assert (len(mlps) == len(nsamples) == len(radii))

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn=bn))

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None, inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, C) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped, inds
        ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)
            new_features = self.mlps[i](
                new_features
            )  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), inds


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if hasattr(m, 'weight') and hasattr(m.weight, 'data'):
            torch.nn.init.constant_(m.weight.data, 0.0)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and hasattr(m.weight, 'data'):
            torch.nn.init.constant_(m.weight.data, 0.0)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            torch.nn.init.constant_(m.bias.data, 0.0)

# class STN3d(nn.Module):
#     def __init__(self, num_points=2500):
#         super(STN3d, self).__init__()
#         self.num_points = num_points
#         self.conv1 = nn.Conv1d(3, 64, 1)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(128, 256, 1)
#         self.mp1 = nn.MaxPool1d(num_points)
#         self.fc1 = nn.Linear(256, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 12)
#         self.relu = nn.ReLU(inplace=True)

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.bn4 = nn.BatchNorm1d(128)
#         self.bn5 = nn.BatchNorm1d(64)

#         self.apply(weights_init)

#     def forward(self, grouped_xyz):
#         device = grouped_xyz.device
#         batch_size, _, N_proposals, _ = grouped_xyz.size()
#         grouped_xyz = grouped_xyz.transpose(2, 1).contiguous().view(batch_size * N_proposals, 3, self.num_points*19)

#         x = self.relu(self.bn1(self.conv1(grouped_xyz)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.mp1(x)
#         x = x.squeeze(2)

#         x = self.relu(self.bn4(self.fc1(x)))
#         x = self.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)

#         iden = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).float().view(1, 12).to(device)

#         x = x + iden
#         x = x.view(batch_size * N_proposals, 3, 4)

#         # coordinates tranformation
#         grouped_xyz = torch.bmm(x[:, :, :3], grouped_xyz) + x[:, :, 3].unsqueeze(-1)
#         grouped_xyz = grouped_xyz.view(batch_size, N_proposals, 3, -1)

#         return grouped_xyz.transpose(1, 2)


# class STN_Group(nn.Module):

#     def __init__(
#             self,
#             radius: float = None,
#             nsample: int = None,
#             use_xyz: bool = True,
#             normalize_xyz: bool = False,  # noramlize local XYZ with radius
#             sample_uniformly: bool = False,
#             ret_unique_cnt: bool = False
#     ):
#         super().__init__()

#         self.radius = radius
#         self.nsample = nsample
#         self.use_xyz = use_xyz
#         self.normalize_xyz = normalize_xyz
#         self.ret_unique_cnt = ret_unique_cnt

#         self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
#                                                      use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
#                                                      sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)

#         self.stn3d = STN3d(num_points=nsample)

#     def forward(self, xyz: torch.Tensor,
#                 features: torch.Tensor = None,
#                 new_xyz: torch.Tensor = None,
#                 orientations: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
#         r"""
#         Parameters
#         ----------
#         xyz : torch.Tensor
#             (B, N, 3) tensor of the xyz coordinates of the features
#         features : torch.Tensor
#             (B, C, N) tensor of the descriptors of the the features
#         new_xyz : torch.Tensor
#             (B, npoint, 3) tensor of the coordinates to be grouped at
#         """
#         if not self.ret_unique_cnt:
#             grouped_features, grouped_xyz = self.grouper(
#                 xyz, new_xyz, features
#             )  # (B, C, npoint, nsample)
#         else:
#             grouped_features, grouped_xyz, unique_cnt = self.grouper(
#                 xyz, new_xyz, features
#             )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)

#         # align objects to the canonical system.
#         rot_matrix = torch.zeros(size=[*orientations.size(), 3, 3]).to(orientations.device)
#         rot_matrix[..., 0, 0] = torch.cos(orientations)
#         rot_matrix[..., 0, 1] = torch.sin(orientations)
#         rot_matrix[..., 1, 1] = torch.cos(orientations)
#         rot_matrix[..., 1, 0] = -torch.sin(orientations)
#         rot_matrix[..., 2, 2] = 1.

#         batch_size, N_proposals = orientations.size()

#         grouped_xyz = torch.bmm(rot_matrix.view(batch_size * N_proposals, 3, 3),
#                                 grouped_xyz.transpose(1, 2).contiguous().view(batch_size * N_proposals, 3, -1))

#         grouped_xyz = grouped_xyz.view(batch_size, N_proposals, 3, -1).transpose(1, 2).contiguous()

#         # Involve STN to learn a spatial transformation (3 X 4 matrix)
#         grouped_xyz = self.stn3d(grouped_xyz)

#         if not self.ret_unique_cnt:
#             return grouped_xyz, grouped_features
#         else:
#             return grouped_xyz, grouped_features, unique_cnt

class STN3d(nn.Module):
    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.mp1 = nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 12)
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

        self.apply(weights_init)

    def forward(self, grouped_xyz):
        device = grouped_xyz.device
        batch_size, N_proposals, _, _ = grouped_xyz.size()
        grouped_xyz = grouped_xyz.view(batch_size * N_proposals, 3, self.num_points)

        x = self.relu(self.bn1(self.conv1(grouped_xyz)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.squeeze(2)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).float().view(1, 12).to(device)

        x = x + iden
        x = x.view(batch_size * N_proposals, 3, 4)

        # coordinates tranformation
        grouped_xyz = torch.bmm(x[:, :, :3], grouped_xyz) + x[:, :, 3].unsqueeze(-1)
        grouped_xyz = grouped_xyz.view(batch_size, N_proposals, 3, -1)

        return grouped_xyz

class STN_GroupSurFeat(nn.Module):

    def __init__(
            self,
            radius: float = None,
            nsample: int = None,
    ):
        super().__init__()

        self.radius = radius
        self.nsample = nsample

        self.sampler = PointnetFPModule([256, 128, 128])

        self.grouper = pointnet2_utils.QueryAndGroupSur(radius, nsample)

        self.stn3d = STN3d(num_points=nsample*19)

    def forward(self, xyz: torch.Tensor,
                feat_xyz: torch.Tensor = None,
                features: torch.Tensor = None,
                new_xyz: torch.Tensor = None,
                center_xyz: torch.Tensor = None,
                orientations: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the coordinates to be grouped at
        """
        B, N_proposals, _ = center_xyz.shape
        grouped_xyz = self.grouper(xyz, new_xyz).view(B, 3, N_proposals, 19, self.nsample)\
            .view(B, 3, N_proposals, -1).transpose(1,2).contiguous() # (B, N_proposal, 3, nsample*19)
        # np.save('grouped_xyz.npy', grouped_xyz.detach().cpu().numpy())
        grouped_features = self.sampler(unknown=grouped_xyz.transpose(2,3).contiguous().view(B, -1, 3),
                                        known=feat_xyz,
                                        unknow_feats=None, 
                                        known_feats=features,
                                        )
        grouped_features = grouped_features.view(B, 128, N_proposals, self.nsample*19)

        grouped_xyz = (grouped_xyz - center_xyz.unsqueeze(-1)) / self.radius
        

        # align objects to the canonical system.
        rot_matrix = torch.zeros(size=[*orientations.size(), 3, 3]).to(orientations.device)
        rot_matrix[..., 0, 0] = torch.cos(orientations)
        rot_matrix[..., 0, 1] = torch.sin(orientations)
        rot_matrix[..., 1, 1] = torch.cos(orientations)
        rot_matrix[..., 1, 0] = -torch.sin(orientations)
        rot_matrix[..., 2, 2] = 1.

        batch_size, N_proposals = orientations.size()

        grouped_xyz = torch.bmm(rot_matrix.view(batch_size * N_proposals, 3, 3),
                                grouped_xyz.view(batch_size * N_proposals, 3, -1))

        grouped_xyz = grouped_xyz.view(batch_size, N_proposals, 3, -1)

        # Involve STN to learn a spatial transformation (3 X 4 matrix)
        grouped_xyz = self.stn3d(grouped_xyz)

        return grouped_xyz, grouped_features

class STN_GroupSur(nn.Module):

    def __init__(
            self,
            radius: float = None,
            nsample: int = None,
    ):
        super().__init__()

        self.radius = radius
        self.nsample = nsample

        self.grouper = pointnet2_utils.QueryAndGroupSur(radius, nsample)

        # self.stn3d = STN3d(num_points=nsample*19)

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor = None):
        grouped_xyzs = []
        B, N_proposals, _, _ = new_xyz.shape
        xyz = torch.cat([xyz, new_xyz.view(B, -1, 3)], dim=1)  # 为防止周围没有点
        for bid in range(B):
            batch_grouped_xyzs = []
            for nid in range(N_proposals):
                grouper = pointnet2_utils.QueryAndGroupSur(round(self.radius[bid][nid].item(), 5), self.nsample)
                grouped_xyz_ = grouper(xyz[bid].unsqueeze(0), new_xyz[bid, nid, :, :].unsqueeze(0))
                batch_grouped_xyzs.append(grouped_xyz_)  # B,3,npoint(18), nsample(9)
            batch_grouped_xyzs = torch.cat(batch_grouped_xyzs, dim=2)  # B,3,KXnpoint(6*18=108), nsample(9)
            grouped_xyzs.append(batch_grouped_xyzs)
        grouped_xyz_ = torch.cat(grouped_xyzs, dim=0)  # (B, 3, N_proposal*18, nsample)
        npoint = new_xyz.shape[2]
        grouped_xyz_ = grouped_xyz_.view(B, 3, N_proposals, npoint, self.nsample) \
            .view(B, 3, N_proposals, -1).transpose(1, 2).contiguous()  # (B, N_proposal, 3, nsample(9)*19(/18))

        return grouped_xyz_

class STN_GroupSurLen(nn.Module):

    def __init__(
            self,
            radius: float = None,
            nsample: int = None,
    ):
        super().__init__()

        self.radius = radius
        self.nsample = nsample

        self.grouper = pointnet2_utils.QueryAndGroupSur(radius, nsample)

        self.stn3d = STN3d(num_points=nsample*19)

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                new_xyz: torch.Tensor = None,
                center_xyz: torch.Tensor = None,
                orientations: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the coordinates to be grouped at
        """
        B, N_proposals, _ = center_xyz.shape
        grouped_xyz = self.grouper(xyz, new_xyz).view(B, 3, N_proposals, 19, self.nsample)\
            .view(B, 3, N_proposals, -1).transpose(1,2).contiguous() # (B, N_proposal, 3, nsample*19)
        # np.save('grouped_xyz.npy', grouped_xyz.detach().cpu().numpy())
        # x = (grouped_xyz[:,:,0,:].max(-1)[0] - grouped_xyz[:,:,0,:].min(-1)[0])
        # y = (grouped_xyz[:,:,1,:].max(-1)[0] - grouped_xyz[:,:,1,:].min(-1)[0])
        # z = (grouped_xyz[:,:,2,:].max(-1)[0] - grouped_xyz[:,:,2,:].min(-1)[0])

        distance = grouped_xyz.pow(2).sum(2).sqrt().max(-1)[0].unsqueeze(-1).unsqueeze(-1).repeat(1,1,3,1)
        # length = torch.sqrt(x**2 + y**2 + z**2)


        grouped_xyz = (grouped_xyz - center_xyz.unsqueeze(-1)) / distance.detach()


        # align objects to the canonical system.
        rot_matrix = torch.zeros(size=[*orientations.size(), 3, 3]).to(orientations.device)
        rot_matrix[..., 0, 0] = torch.cos(orientations)
        rot_matrix[..., 0, 1] = torch.sin(orientations)
        rot_matrix[..., 1, 1] = torch.cos(orientations)
        rot_matrix[..., 1, 0] = -torch.sin(orientations)
        rot_matrix[..., 2, 2] = 1.

        batch_size, N_proposals = orientations.size()

        grouped_xyz = torch.bmm(rot_matrix.view(batch_size * N_proposals, 3, 3),
                                grouped_xyz.view(batch_size * N_proposals, 3, -1))

        grouped_xyz = grouped_xyz.view(batch_size, N_proposals, 3, -1)

        # Involve STN to learn a spatial transformation (3 X 4 matrix)
        grouped_xyz = self.stn3d(grouped_xyz)

        return grouped_xyz

class STN_Group(nn.Module):

    def __init__(
            self,
            radius: float = None,
            nsample: int = None,
            use_xyz: bool = True,
            normalize_xyz: bool = False,  # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False
    ):
        super().__init__()

        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
                                                     use_xyz=use_xyz, normalize_xyz=normalize_xyz,
                                                     sample_uniformly=sample_uniformly)

        self.stn3d = STN3d_original(num_points=nsample)

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                new_xyz: torch.Tensor = None,
                orientations: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the coordinates to be grouped at
        """

        groupe_results = self.grouper(xyz, new_xyz, features) # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)
        grouped_features = groupe_results['new_features']
        grouped_xyz = groupe_results['grouped_xyz']


        # align objects to the canonical system.
        rot_matrix = torch.zeros(size=[*orientations.size(), 3, 3]).to(orientations.device)
        rot_matrix[..., 0, 0] = torch.cos(orientations)
        rot_matrix[..., 0, 1] = torch.sin(orientations)
        rot_matrix[..., 1, 1] = torch.cos(orientations)
        rot_matrix[..., 1, 0] = -torch.sin(orientations)
        rot_matrix[..., 2, 2] = 1.

        batch_size, N_proposals = orientations.size()

        grouped_xyz = torch.bmm(rot_matrix.view(batch_size * N_proposals, 3, 3),
                                grouped_xyz.transpose(1, 2).contiguous().view(batch_size * N_proposals, 3, -1))

        grouped_xyz = grouped_xyz.view(batch_size, N_proposals, 3, -1).transpose(1, 2).contiguous()

        # Involve STN to learn a spatial transformation (3 X 4 matrix)
        grouped_xyz = self.stn3d(grouped_xyz)
        return grouped_xyz, grouped_features

class STN3d_original(nn.Module):
    def __init__(self, num_points=2500):
        super(STN3d_original, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.mp1 = nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 12)
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

        self.apply(weights_init)

    def forward(self, grouped_xyz):
        device = grouped_xyz.device
        batch_size, _, N_proposals, _ = grouped_xyz.size()
        grouped_xyz = grouped_xyz.transpose(2, 1).contiguous().view(batch_size * N_proposals, 3, self.num_points)

        x = self.relu(self.bn1(self.conv1(grouped_xyz)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.squeeze(2)

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).float().view(1, 12).to(device)

        x = x + iden
        x = x.view(batch_size * N_proposals, 3, 4)

        # coordinates tranformation
        grouped_xyz = torch.bmm(x[:, :, :3], grouped_xyz) + x[:, :, 3].unsqueeze(-1)
        grouped_xyz = grouped_xyz.view(batch_size, N_proposals, 3, -1)

        return grouped_xyz.transpose(1, 2)


if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(
            torch.cuda.FloatTensor(*new_features.size()).fill_(1)
        )
        print(new_features)
        print(xyz.grad)