import torch
import torch.nn as nn
from timm.models.layers import DropPath,trunc_normal_
import numpy as np
from external.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from models.registers import MODULES
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import build_shared_mlp1d
# from knn_cuda import KNN
# knn = KNN(k=8, transpose_mode=False)

def knn_point(nsample, xyz, new_xyz, return_values=False):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    knn_values, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    if return_values:
        return knn_values, group_idx
    else:
        return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist   

def get_knn_index(coor_q, coor_k=None):
    coor_k = coor_k if coor_k is not None else coor_q
    # coor: bs, 3, np
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2)

    with torch.no_grad():
#         _, idx = knn(coor_k, coor_q)  # bs k np
        idx = knn_point(8, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
        idx = idx.transpose(-1, -2).contiguous()
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)
    
    return idx  # bs*k*np

def get_graph_feature(x, knn_index, x_q=None):

        #x: bs, np, c, knn_index: bs*k*np
        k = 8
        batch_size, num_points, num_dims = x.size()
        num_query = x_q.size(1) if x_q is not None else num_points
        feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
        feature = feature.view(batch_size, k, num_query, num_dims)
        x = x_q if x_q is not None else x
        x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1)
        feature = torch.cat((feature - x, x), dim=-1)
        return feature  # b k np c


class DGCNN_Grouper(nn.Module):
    def __init__(self, input_dim = 3, hidden_dim = 8):
        super(DGCNN_Grouper,self).__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(input_dim, hidden_dim, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(hidden_dim*2, 32, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 32),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        # coor: bs, 3, np, x: bs, c, np

        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            #             _, idx = knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous())  # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, downsample_point_numbers=[512, 128]):
        # x: bs, 3, np

        # bs 3 N(128)   bs C(224)128 N(128)
        coor = x
        f = self.input_trans(x)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, downsample_point_numbers[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, downsample_point_numbers[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        return coor, f


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.knn_map_cross = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map_cross = nn.Linear(dim*2, dim)

    def forward(self, q, v, self_knn_index=None, cross_knn_index=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)

        if self_knn_index is not None:
            knn_f = get_graph_feature(norm_q, self_knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_1 = torch.cat([q_1, knn_f], dim=-1)
            q_1 = self.merge_map(q_1)
        
        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        if cross_knn_index is not None:
            knn_f = get_graph_feature(norm_v, cross_knn_index, norm_q)
            knn_f = self.knn_map_cross(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_2 = torch.cat([q_2, knn_f], dim=-1)
            q_2 = self.merge_map_cross(q_2)

        q = q + self.drop_path(q_2)

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, knn_index = None):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        norm_x = self.norm1(x)
        x_1 = self.attn(norm_x)

        if knn_index is not None:
            knn_f = get_graph_feature(norm_x, knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            x_1 = torch.cat([x_1, knn_f], dim=-1)
            x_1 = self.merge_map(x_1)
        
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

@MODULES.register_module
class PCTransformer_small(nn.Module):
    """ Vision Transformer with support for point cloud completion
    """
    def __init__(self, config, optim_spec):
        super(PCTransformer_small, self).__init__()
        in_chans = 3
        embed_dim = config.get('embed_dim', 768)
        #下采样两层，顶点attention三层
        depth = config.get('depth', 3)
        num_heads = config.get('num_heads', 3)
        mlp_ratio = config.get('mlp_ratio', 2)
        c_dim = config['output_cdim']
        qkv_bias = False,
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        knn_layer = -1
        use_rgb_features = config.get('use_rgb_features',False)
        feat_dim = 256 * config['use_proposal_features'] + 256 * use_rgb_features
        self.num_features = self.embed_dim = embed_dim

        self.knn_layer = knn_layer

        self.grouper = DGCNN_Grouper()  # B 3 N to B C(3) N(128) and B C(128) N(128)

        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )

        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        self.encoder = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])

        self.feat_fusion = nn.Sequential(
            nn.Linear(embed_dim + feat_dim, embed_dim), #768+512 -> 256
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim//2, c_dim)
        )
        # self.increase_dim = nn.Sequential(
        #     nn.Conv1d(embed_dim, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(1024, 1024, 1)
        # )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, inpc, feat=None):
        '''
            inpc : input incomplete point cloud with shape B N(2048) C(3)
        '''
        coor, f = self.grouper(inpc.transpose(1, 2).contiguous(), downsample_point_numbers=[256,128])
        knn_index = get_knn_index(coor)
        pos = self.pos_embed(coor).transpose(1, 2)
        x = self.input_proj(f).transpose(1, 2)

        for i, blk in enumerate(self.encoder):
            if i < self.knn_layer:
                x = blk(x + pos, knn_index)  # B N C
            else:
                x = blk(x + pos)

        x_global = x.mean(1) #B,C
        if feat is not None:
            feat = torch.cat([x_global, feat],-1)
        else:
            feat = x_global
        output_feat = self.feat_fusion(feat)
        return output_feat


class PCTransformer(nn.Module):
    """ Vision Transformer with support for point cloud completion
    """
    def __init__(self, in_chans=3, embed_dim=768, depth=6, num_heads=6, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,knn_layer=-1):
        super(PCTransformer, self).__init__()

        self.num_features = self.embed_dim = embed_dim

        self.knn_layer = knn_layer

        self.grouper = DGCNN_Grouper()  # B 3 N to B C(3) N(128) and B C(128) N(128)

        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )

        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        self.encoder = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])


        # self.increase_dim = nn.Sequential(
        #     nn.Conv1d(embed_dim, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(1024, 1024, 1)
        # )


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64  #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda()
        freqs = np.pi * (2 ** freqs)

        freqs = freqs.view(*[1] * len(normal_coor.shape), -1)  # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1)  # B x 3 x N x 1
        k = normal_coor * freqs  # B x 3 x N x D
        s = torch.sin(k)  # B x 3 x N x D
        c = torch.cos(k)  # B x 3 x N x D
        x = torch.cat([s, c], -1)  # B x 3 x N x 2D
        pos = x.transpose(-1, -2).reshape(coor.shape[0], -1, coor.shape[-1])  # B 6D N

        return pos

    def forward(self, inpc, downsample_point_numbers):
        '''
            inpc : input incomplete point cloud with shape B N(2048) C(3)
        '''
        coor, f = self.grouper(inpc.transpose(1, 2).contiguous(), downsample_point_numbers)
        knn_index = get_knn_index(coor)
        pos = self.pos_embed(coor).transpose(1, 2)
        x = self.input_proj(f).transpose(1, 2)

        for i, blk in enumerate(self.encoder):
            if i < self.knn_layer:
                x = blk(x + pos, knn_index)  # B N C
            else:
                x = blk(x + pos)


        # global_feature = self.increase_dim(x.transpose(1, 2))  # B 1024 N

        return x.transpose(1, 2), coor.transpose(1, 2)

class DGCNN_Grouper_with_feature(nn.Module):
    def __init__(self, coor_adjust_layers = [3,16], f_adjust_layers=[512,128], hidden_dim = 64):
        super(DGCNN_Grouper_with_feature,self).__init__()
        '''
        K has to be 16
        '''
        self.x_adjust = build_shared_mlp1d(coor_adjust_layers)
        self.f_adjust = build_shared_mlp1d(f_adjust_layers)
        fusion_layers = [coor_adjust_layers[-1]+f_adjust_layers[-1], hidden_dim, hidden_dim]
        self.combine = build_shared_mlp1d(fusion_layers)

        self.layer1 = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, hidden_dim),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim,kernel_size=1, bias=False),
                                    nn.GroupNorm(4, hidden_dim),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, hidden_dim),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, hidden_dim*2),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        # coor: bs, 3, np, x: bs, c, np

        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            #             _, idx = knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous())  # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, f, downsample_point_numbers=[512, 128]):
        coor = x
        x_f = self.x_adjust(x) # 3-> 8
        f_ = self.f_adjust(f)  # 128->xx
        f = self.combine(torch.cat([x_f, f_],dim=1))#xx+xx -> hidden_dim


        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, downsample_point_numbers[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, downsample_point_numbers[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        return coor, f

@MODULES.register_module
class PCTransformer_with_feature(nn.Module):
    """ Vision Transformer with support for point cloud completion
    """
    def __init__(self, config, optim_spec):
        super(PCTransformer_with_feature, self).__init__()
        in_chans = 3
        embed_dim = config.get('embed_dim',192)  # 768 192/32x6
        depth = config.get('depth',6)
        num_heads = 6
        mlp_ratio = 2.
        qkv_bias = False,
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        knn_layer = -1
        coor_adjust_layers = config.get('coor_adjust_layers',[3, 16])
        f_adjust_layers = config.get('f_adjust_layers',[512, 128])
        dgcnn_hidden_dim =  config.get('dgcnn_hidden_dim',64)

        self.num_features = self.embed_dim = embed_dim

        self.knn_layer = knn_layer

        self.grouper = DGCNN_Grouper_with_feature( coor_adjust_layers, f_adjust_layers, dgcnn_hidden_dim)  # B 3 N to B C(3) N(128) and B C(128) N(128)

        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )

        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        self.encoder = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])


        # self.increase_dim = nn.Sequential(
        #     nn.Conv1d(embed_dim, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(1024, 1024, 1)
        # )


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64  #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda()
        freqs = np.pi * (2 ** freqs)

        freqs = freqs.view(*[1] * len(normal_coor.shape), -1)  # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1)  # B x 3 x N x 1
        k = normal_coor * freqs  # B x 3 x N x D
        s = torch.sin(k)  # B x 3 x N x D
        c = torch.cos(k)  # B x 3 x N x D
        x = torch.cat([s, c], -1)  # B x 3 x N x 2D
        pos = x.transpose(-1, -2).reshape(coor.shape[0], -1, coor.shape[-1])  # B 6D N

        return pos

    def forward(self, inpc, inpfeat, downsample_point_numbers):
        '''
            inpc : input incomplete point cloud with shape B N(2048) C(3)
        '''
        coor, f = self.grouper(inpc.transpose(1, 2).contiguous(),  inpfeat, downsample_point_numbers)
        knn_index = get_knn_index(coor)
        pos = self.pos_embed(coor).transpose(1, 2)
        x = self.input_proj(f).transpose(1, 2)

        for i, blk in enumerate(self.encoder):
            if i < self.knn_layer:
                x = blk(x + pos, knn_index)  # B N C
            else:
                x = blk(x + pos)


        # global_feature = self.increase_dim(x.transpose(1, 2))  # B 1024 N

        return x.transpose(1, 2), coor.transpose(1, 2)

@MODULES.register_module
class PointEncoder(nn.Module):
    def __init__(self, config, optim_spec):
        super(PointEncoder, self).__init__()
        self.pc_dim = config['pc_dim']
        self.knn_layer = config['knn_layer']
        self.fusion_dim = config['fusion_dim']
        # d_attn = config.fusion_attn_dim
        # num_heads = config['fusion_num_heads']
        self.use_proposal_feature = config['use_proposal_features']
        self.proposal_feature_dim = config['proposal_feature_dim']
        self.latent_feature_dim = config['latent_feature_dim']
        encoder_depth = 6

        # Encoders for images and Point clouds
        self.pc_encoder = PCTransformer(embed_dim = self.pc_dim, num_heads=6, depth = encoder_depth,
                                               knn_layer = self.knn_layer)

        self.pc_adjust_dim = nn.Sequential(
            nn.Conv1d(self.pc_dim, self.fusion_dim, 1),
            nn.BatchNorm1d(self.fusion_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.fusion_dim, self.fusion_dim, 1)
        )

        if self.use_proposal_feature:
            self.feature_combine = nn.Sequential(
                nn.Linear(self.fusion_dim + self.proposal_feature_dim, 512),
                # nn.BatchNorm1d(1024),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(512, self.latent_feature_dim)
            )
        self.apply(self._init_weights)
        # self.build_loss_func()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, input_points, proposal_feature=None):
        pc_feature, coor = self.pc_encoder(input_points, downsample_point_numbers = [256,128]) #B,C,N
        adjusted_pc_feature =self.pc_adjust_dim(pc_feature) #B,C,N
        global_feature = torch.max(adjusted_pc_feature , dim=-1)[0] #B,C
        if self.use_proposal_feature and proposal_feature is not None:
            global_feature_ = torch.cat([global_feature, proposal_feature],dim=1)
            global_feature = self.feature_combine(global_feature_)
        return global_feature