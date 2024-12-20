import torch
import torch.nn as nn
import warnings
from torch.autograd import Function
from typing import *
import numpy as np
try:
    import pointnet2_ops._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os

    warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")

    _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
    _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
        osp.join(_ext_src_root, "src", "*.cu")
    )
    _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[osp.join(_ext_src_root, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = _ext.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        ctx.save_for_backward(idx, features)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        ctx.save_for_backward(idx, weight, features)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        ctx.save_for_backward(idx, features)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, torch.zeros_like(idx)


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        output = _ext.ball_query(new_xyz, xyz, radius, nsample)

        ctx.mark_non_differentiable(output)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


ball_query = BallQuery.apply

def QueryFromAnchors(xyz, anchors, dist, nsample = [30,5], iterations = 2, output_points_num = 256,
                     min_num_pts = 20, return_sampling_number_per_anchor = False):
    batch_size, Nproposal, Nanchor,_ = anchors.shape
    output_xyz = torch.empty([batch_size, Nproposal, output_points_num, 3])

    non_empty_mask = np.ones((batch_size, Nproposal))
    number_of_query_per_anchor = np.zeros((batch_size, Nproposal, Nanchor),np.uint8)
    sampled_point_indices = np.zeros((batch_size, Nproposal, output_points_num),np.uint64)
    pc_nums = []
    for bid in range(batch_size):
        # batch_xyz = xyz[bid][xyz[bid, :, 2] > floor_height[bid]] if floor_height is not None else xyz[bid]
        batch_xyz = xyz[bid]
        pc_nums.append(batch_xyz.shape[0])
        for nid in range(Nproposal):
            idxs = []
            radius = dist[bid][nid]
            batch_anchors = anchors[bid][nid]
            xyz_ = torch.cat([batch_xyz, batch_anchors], dim=0)# (18 + 80000) ,3
            for iter in range(iterations):
                idx = ball_query(radius/(2**iter), nsample[iter], xyz_.unsqueeze(0), batch_anchors.unsqueeze(0))   #B, nanchor, nsample
                if iter == 0:
                    for anchor_id in range(idx.shape[1]):
                        number_of_query_per_anchor[bid][nid][anchor_id] = len(torch.unique(idx[0][anchor_id]))-1
                xyz_trans = xyz_.t().contiguous().unsqueeze(0) #1,3,(18+800000)
                batch_anchors = grouping_operation(xyz_trans, idx).permute(0,2,3,1).contiguous().view(-1,3)# (B, 3, npoint, nsample) ->(B, npoint, nsample, 3)
                idxs.append(idx.flatten()) #B, Nproposal, 18 * nsample

            idxs = torch.cat(idxs)
            unique_ind = torch.unique(idxs) #N
            if len(unique_ind) < min_num_pts:
                non_empty_mask[bid][nid]=0
            temp_xyz = xyz_[unique_ind.long()].unsqueeze(0) #unique_n, 3
            temp_xyz_flipped = temp_xyz.transpose(1, 2).contiguous()
            fps_id = furthest_point_sample(temp_xyz, output_points_num) #1, 1024
            output_xyz[bid][nid] = gather_operation(temp_xyz_flipped,fps_id).transpose(1, 2).contiguous()[0]
            sampled_point_indices[bid][nid] = unique_ind[fps_id[0].long()].cpu().numpy()
    if return_sampling_number_per_anchor:
        return output_xyz.to(xyz.device), non_empty_mask.astype(bool),number_of_query_per_anchor, sampled_point_indices, np.array(pc_nums)
    else:
        return output_xyz.to(xyz.device), non_empty_mask.astype(bool)


def QueryFromAnchors_parallel(xyz, anchors, radius=0.1, nsample = [30,5], iterations = 2):
    #是否允许重复采样?
    # 目的： 找到出现点次数最多的instance id
    batch_size, Nproposal, Nanchor,_ = anchors.shape
    idxs = []
    anchors_ = anchors.view(batch_size, -1, 3).contiguous() #B,Nproposal*Nanchor,3
    xyz_ = torch.cat([xyz, anchors_], dim=1) #B,N+Nproposal*Nsample,3
    for iter in range(iterations):
        idx = ball_query(radius / (2 ** iter), nsample[iter], xyz_, anchors_)  #B,Nproposal*Nanchor, nsample
        xyz_trans = xyz_.permute(0,2,1).contiguous()  # B,3,N+Nproposal*Nanchor
        anchors_ = grouping_operation(xyz_trans, idx).permute(0, 2, 3, 1).contiguous().view(batch_size,-1,3)
        # (B, 3, Nproposal*Nanchor, nsample) ->(B, Nproposal*Nanchor, nsample, 3) -> B,-1,3
        idxs.append(idx.view(batch_size, Nproposal, -1).contiguous())
        #[B,Nproposal, nsample[0]*Nanchor, B,Nproposal,nsample[1]*Nanchor*nsample[0]]

    #fps: B,N,3 -> B,Npoint

    sampled_point_indices = torch.cat(idxs,dim=-1) #B,Nproposal,xxx

    return sampled_point_indices

def QueryFromAnchors_per_proposal(xyz, anchors, dist, nsample = [30,5], iterations = 2, return_xyz=False, output_points_num = 480):
    center = (torch.max(anchors,0)[0] +torch.min(anchors,0)[0])/2.
    max_dist = torch.abs(anchors - center.unsqueeze(0)).max(0)[0] + torch.sqrt(dist).repeat(3)#3
    in_search_ids = (torch.abs(xyz - center.unsqueeze(0)) < max_dist).all(1)
    # max_dist = torch.sum(((anchors - center.unsqueeze(0))**2),1).max() + dist #anchors: 18,3; center: 3 -> 1
    # in_search_ids = torch.sum(((xyz - center.unsqueeze(0))**2),1) < max_dist
    in_search_pc = xyz[in_search_ids] #80000, 3 -> N,3
    n_point = in_search_ids.sum()
    xyz_ = torch.cat([in_search_pc, anchors], dim=0)  # (18 + 80000) ,3
    idxs = []
    for iter in range(iterations):
        idx = ball_query(dist / (2 ** iter), nsample[iter], xyz_.unsqueeze(0),
                         anchors.unsqueeze(0))  # B, nanchor, nsample
        xyz_trans = xyz_.t().contiguous().unsqueeze(0)  # 1,3,(18+800000)
        anchors = grouping_operation(xyz_trans, idx).permute(0, 2, 3, 1).contiguous().view(-1,3)  # (B, 3, npoint, nsample) ->(B, npoint, nsample, 3)
        idxs.append(idx.flatten())  # B, Nproposal, 18 * nsample
    idxs = torch.cat(idxs)
    unique_ind = torch.unique(idxs)  # N
    # unique_xyz = xyz_[unique_ind.long()]  # unique_n, 3
    unique_ind_ = unique_ind[unique_ind<n_point]
    original_unique_ind = torch.where(in_search_ids)[0][unique_ind_.long()]

    if return_xyz:
        #包含anchor在内的点都参与采样
        unique_sampled_xyz = xyz_[unique_ind.long()]
        if len(unique_sampled_xyz) > output_points_num:
            intra_sampled_id = torch.randperm(len(unique_sampled_xyz))[:output_points_num]
        else:
            extra_pts_num = output_points_num - len(unique_sampled_xyz)  # 先sampled一遍，再sampled一遍
            intra_sampled_id = torch.cat([torch.arange(len(unique_sampled_xyz)),
                                          torch.randint(len(unique_sampled_xyz), (extra_pts_num,))])

        sampled_xyzs = unique_sampled_xyz[intra_sampled_id]
        return original_unique_ind, sampled_xyzs
    else:
        return original_unique_ind



class QueryAndGroupSur(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample):
        super(QueryAndGroupSur, self).__init__()
        self.radius, self.nsample = radius, nsample

    def forward(self, xyz, new_xyz):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        Returns
        -------
        grouped_xyz : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
        for i_batch in range(idx.shape[0]):
            for i_region in range(idx.shape[1]):
                unique_ind = torch.unique(idx[i_batch, i_region, :])
                num_unique = unique_ind.shape[0]
                unique_cnt[i_batch, i_region] = num_unique
                sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                idx[i_batch, i_region, :] = all_ind


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)

        return grouped_xyz
        

class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, normalize_xyz=False, sample_uniformly=False):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        ret = {}
        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind
            ret['unique_cnt'] = unique_cnt


        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz_pre = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz = grouped_xyz_pre - new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz


        ret['new_features'] = new_features
        ret['grouped_xyz'] = grouped_xyz
        ret['grouped_xyz_pre']=grouped_xyz_pre
        ret['idx'] = idx
        return ret


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features
