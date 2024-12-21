import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registers import LOSSES
from net_utils.nn_distance import nn_distance, huber_loss, smoothl1_loss, nn_distance_self
from net_utils.ap_helper import parse_groundtruths, get_pred_bbox, get_gt_bbox
# from net_utils.quad_ap_helper import get_quad_bbox, parse_quad_groundtruths
from net_utils.box_util import box3d_iou, get_3d_box_cuda
from net_utils.libs import softmax,extract_pc_in_box3d, sigmoid
from utils.pc_util import write_ply, write_ply_color
from external.pointnet2_ops_lib.pointnet2_ops import pointnet2_modules, pointnet2_utils
from net_utils.quad_ap_helper import LENGTH as WIDTH
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import QueryFromAnchors_per_proposal, \
    QueryFromAnchors_parallel
# from models.anchorrec.CAGroup.PCTransformer import knn_point
from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance
from sklearn.neighbors import NearestNeighbors
chamfer_func = ChamferDistance()

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
QUAD_NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness
QUAD_CLS_WEIGHTS = [0.4, 0.6]

criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
objectness_criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
criterion_size_class = nn.CrossEntropyLoss(reduction='none')
criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
criterion_mse_loss = nn.MSELoss(reduction = 'mean')
rgb_classify_criterion = nn.CrossEntropyLoss(reduction='none')
# '''for soft objectness loss'''
rgb_objectness_criterion = nn.CrossEntropyLoss(reduction='none')
# rgb_objectness_criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
# rgb_objectness_criterion = nn.BCEWithLogitsLoss()
class BaseLoss(object):
    '''base loss class'''

    def __init__(self, cfg, weight=1):
        '''initialize loss module'''
        self.iter = 0
        self.vis_epoch = 1
        self.cfg = cfg
        self.weight = weight


@LOSSES.register_module
class Null(BaseLoss):
    '''This loss function is for modules where a loss preliminary calculated.'''

    def __call__(self, loss):
        return self.weight * torch.mean(loss)


def SoftCrossEntropyLoss(p, labels, weights=None):
    '''
    Args:
        predicted: B,num_class,Nproposal
        smoothed_labels: B,num_class,Nproposal (smoothed one-hot)
        weights: np.array(num class)
    Returns:

    '''
    positive_ratios = torch.sum(labels, 1) / labels.shape[1]
    # 如果没有positive的结果，就把positive ratio设为一个极小值0.02
    positive_ratios[torch.where(positive_ratios == 0)] = 0.02
    mask = labels.float()
    for bid in range(mask.shape[0]):
        mask[bid, mask[bid] == 1] = 1 - positive_ratios[bid]
        mask[bid, mask[bid] == 0] = positive_ratios[bid]
    # ratio  = (1-positive_ratios) / positive_ratios#has to be a Tensor of size nbatch.
    # ratio  = (1-positive_ratios) / (positive_ratios + 1e-4) #has to be a Tensor of size nbatch.
    # ratio = 4
    # loss = -(weights * labels * torch.log(p) * ratio + (1-labels) * torch.log((1-p)))  # B, 2, nproposal
    # loss = -(weights * labels * torch.log(p) * ratio.unsqueeze(1) + (1-labels) * torch.log((1-p)))  # B, 2, nproposal
    # loss = F.binary_cross_entropy(p,labels*weights,ratio.unsqueeze(1).repeat(1,p.shape[1]))
    if weights == None:
        loss = F.binary_cross_entropy(p, labels, mask)
    else:
        loss = F.binary_cross_entropy(p, labels * weights, mask)
    return loss



def soft_cross_entropy(input, target, reduction='mean'):
    logprobs = F.log_softmax(input, dim=1)
    loss = -torch.sum(target * logprobs, dim=1)

    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return loss

def compute_vote_loss(est_data, gt_data):
    scene_points = gt_data['point_clouds'][...,:3]
    gt_votes = gt_data['vote_label'][...,:3]
    gt_vote_mask = gt_data['vote_label_mask'].bool()
    center_label = scene_points + gt_votes
    vote_xyz = est_data['vote_xyz']
    seed_indices = est_data['seed_indices'].long()
    target_masks = torch.gather(gt_vote_mask, 1, seed_indices)
    est_data['vote_masks'] = target_masks
    if target_masks.sum()==0:
        print('error: %s')
    selected_center_label = torch.gather(center_label, 1, seed_indices.unsqueeze(-1).repeat(1,1,3)) #B,1024,3
    votes_dist = torch.sum(torch.abs(vote_xyz - selected_center_label), dim=-1)  # (B*num_seed,vote_factor) to (B*num_seed,)
    vote_loss = torch.sum(votes_dist * target_masks.float()) / (torch.sum(target_masks.float()) + 1e-6)
    return vote_loss

def compute_quad_vote_loss(est_data, gt_data):
    # 没有vote mask是因为墙面的中心不可能是0,0,0
    # Load ground truth votes and assign them to seed points
    # use_quad = gt_data['use_quad']
    has_quad_ind = gt_data['use_quad'].bool()
    if len(gt_data['use_quad'][gt_data['use_quad'] > 0]) == 0:
        return torch.tensor(0)

    # batch_size = est_data['seed_xyz_bg'][has_quad_ind, :, :].shape[0]
    # num_seed = est_data['seed_xyz_bg'].shape[1]  # B,num_seed,3
    # seed_inds = est_data['seed_indices_bg'][has_quad_ind, :].long()  # B,num_seed in [0,num_points-1]

    batch_size = est_data['seed_xyz'][has_quad_ind, :, :].shape[0]
    num_seed = est_data['seed_xyz'].shape[1]  # B,num_seed,3
    seed_inds = est_data['seed_indices'][has_quad_ind, :].long()  # B,num_seed in [0,num_points-1]

    vote_xyz = est_data['vote_quad_xyz'][has_quad_ind, :, :]  # B,num_seed*vote_factor,3

    seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3)
    seed_gt_votes = torch.gather(gt_data['quad_vote_label'][has_quad_ind, :, :], 1, seed_inds_expand)
    seed_gt_votes_mask = torch.zeros(seed_gt_votes[:, :, 0].shape).to(seed_gt_votes.device)
    indices = torch.stack(torch.where(torch.sum(seed_gt_votes, axis=2) != 0)).t()
    seed_gt_votes_mask[indices[:, 0], indices[:, 1]] = 1

    loss = torch.linalg.norm((vote_xyz - seed_gt_votes).float(), dim=2)  # B,Nseed,
    vote_loss = torch.sum(loss * seed_gt_votes_mask.float()) / (torch.sum(seed_gt_votes_mask.float()) + 1e-6)
    return vote_loss


def compute_objectness_loss(est_data, gt_data,dataset_config, valid_mask=None):
    aggregated_vote_xyz = est_data['aggregated_vote_xyz']
    gt_center = gt_data['center_label'][:, :, 0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    # K2 = gt_center.shape[1]
    num_gt_objects = gt_data['num_gt_objects']
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: BxK, dist2: BxK2

    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    assignment = ind1  # (B,K) with values in 0,1,...,K2-1
    
    objectness_label = torch.zeros((B, K), dtype=torch.long).cuda()
    objectness_label[ind1 >= num_gt_objects] = 0
    # objectness_label[ind1 >= num_gt_objects[:,0]] = 0

    gt_centers = torch.gather(gt_data['center_label'], 1, assignment.unsqueeze(-1).repeat(1, 1, 3))
    parsed_gts = parse_groundtruths(gt_data, dataset_config)
    gt_headings = torch.gather(torch.tensor(parsed_gts['heading_angles']).to(assignment.device), 1, assignment.unsqueeze(-1))
    gt_sizes = torch.gather(torch.tensor(parsed_gts['box_sizes']).to(assignment.device), 1, assignment.unsqueeze(-1).repeat(1, 1, 3))

    canonical_xyz = aggregated_vote_xyz - gt_centers
    from net_utils.box_util import rotation_3d_in_axis
    canonical_xyz = rotation_3d_in_axis(
                canonical_xyz.unsqueeze(2),
                - gt_headings.squeeze(-1), 2).squeeze(2)
    #einsum('aij,jka->aik', (points, rot_mat_T)) rot_mat_T:  # 256,1,3 ;3,3,256 -> 256,1,3
    #einsum('baij,jkba->baik', (points, rot_mat_T)) rot_mat_T:  # 8,256,1,3 ;3,3,8,256  -> 8, 256,1,3

    distance_front = gt_sizes[:, :, 0] - canonical_xyz[:, :, 0]
    distance_left = gt_sizes[:, :, 1] - canonical_xyz[:, :, 1]
    distance_top = gt_sizes[:,:,  2] - canonical_xyz[:, :, 2]
    distance_back = gt_sizes[:, :, 0] + canonical_xyz[:, :,  0]
    distance_right = gt_sizes[:,:,  1] + canonical_xyz[:, :, 1]
    distance_bottom = gt_sizes[:,:,  2] + canonical_xyz[:, :, 2]

    distance_targets = torch.cat(
        (distance_front.unsqueeze(-1),
         distance_left.unsqueeze(-1),
         distance_top.unsqueeze(-1),
         distance_back.unsqueeze(-1),
         distance_right.unsqueeze(-1),
         distance_bottom.unsqueeze(-1)),
        dim=-1
    )# 8,256,6
    inside_mask = (distance_targets >= 0.).all(dim=-1)
    pos_mask =  (euclidean_dist1 < NEAR_THRESHOLD) & inside_mask
    objectness_label[pos_mask]=1

    objectness_scores = est_data['objectness_scores']
    objectness_loss = objectness_criterion(objectness_scores.transpose(2, 1), objectness_label)
    if valid_mask !=None:
        objectness_label2 = objectness_label.clone()
        objectness_label2[~valid_mask] = 0
        objectness_label = objectness_label2
        valid_mask = valid_mask.float()
        objectness_loss = torch.sum(objectness_loss * valid_mask.detach()) / (torch.sum(valid_mask.detach())+ 1e-6)
    else:
        objectness_loss =  objectness_loss.mean()

    # compute accuracy and recall
    obj_logits = objectness_scores.clone().squeeze(2).detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    num_correct = 0
    for batchid in range(objectness_label.shape[0]):
        num_correct += len(np.intersect1d(torch.where(objectness_label[batchid] > 0)[0].cpu().numpy(),
                                          np.where(obj_prob[batchid] > 0.5)[0]))
    objectness_recall = num_correct / torch.sum(objectness_label).item() if torch.sum(
        objectness_label).item() != 0 else 0
    objectness_accuracy = num_correct / len(np.where(obj_prob > 0.5)[1]) if len(
        np.where(obj_prob > 0.5)[1]) != 0 else 0

    return objectness_loss, objectness_label, assignment, objectness_recall, objectness_accuracy


def compute_anchor_loss(est_data, gt_data, cfg, flag=None):
    if flag == 'quad':
        use_quad = gt_data['use_quad']
        has_quad_ind = gt_data['use_quad'].bool()
        if len(gt_data['use_quad'][gt_data['use_quad'] > 0]) == 0:
            return torch.tensor(0)
        object_assignment = est_data['quad_assignment']
        objectness_label = est_data['quad_label']
        anchor = est_data['quad_surface_points']  # batchsize, n, _
        gt_anchor = gt_data['quad_surface_points']
        gt_mask = gt_data['target_quad_mask']
    else:
        object_assignment = est_data['object_assignment']
        objectness_label = est_data['objectness_label']
        anchor = est_data['anchors']
        gt_anchor = gt_data['surface_points']
        # gt_mask = gt_data['target_object_mask']
    avg_dist = torch.tensor(0.0).to(objectness_label.device)
    shape_num = len(anchor)
    bs = anchor[-1].shape[0]

    iter = 0
    for batchid in range(bs):
        if flag == 'quad' and not has_quad_ind[batchid]:
            continue
        indd = torch.nonzero(objectness_label[iter])  # nproposal -> n
        if len(indd) == 0:
            iter += 1
            continue
        indd = indd.squeeze(1)
        surface_points = gt_anchor[batchid]
        if flag == 'quad':
            mask = gt_mask[batchid, object_assignment[iter][indd]]  # 防止有一些前面没有surface points

        for i in range(shape_num):
            batch_anchor = anchor[i][batchid]
            dist1, dist2 = chamfer_func(batch_anchor[indd],  # M,18,3
                                        surface_points[object_assignment[iter][indd]])
            # pc_util.write_ply(surface_points[object_assignment[batchid][indd]].detach().cpu().numpy().reshape(-1,
            #         3), os.path.join('gtvote.ply'))  # surface_points[object_assignment[batchid][indd]]
            if flag == 'quad':
                mask1_reshaped = mask.unsqueeze(1).repeat(1, dist1.shape[1])
                mask2_reshaped = mask.unsqueeze(1).repeat(1, dist2.shape[1])
                dist1 = torch.sum(dist1 * mask1_reshaped) / (torch.sum(mask1_reshaped) + 1e-6)
                dist2 = torch.sum(dist2 * mask2_reshaped) / (torch.sum(mask2_reshaped) + 1e-6)
                avg_dist += cfg['model']['quad_detection']['shape']['wd1'] * dist1 + \
                            cfg['model']['quad_detection']['shape']['wd2'] * dist2
            else:
                avg_dist += cfg['model']['detection']['shape']['wd1'] * torch.mean(dist1) + \
                            cfg['model']['detection']['shape']['wd2'] * torch.mean(dist2)

            # if it % cfg['log']['vis_step'] == 0:
            #     # np.save(os.path.join(cfg['log']['path'], 'visualization','svote_ep{}_it{}.npy'.format(epoch, it)),
            #     #         batch_anchor[indd].detach().cpu().numpy())
            #     # np.save(os.path.join(cfg['log']['path'], 'visualization', 'obj_ep{}_it{}.npy'.format(epoch, it)),
            #     #         surface_points[object_assignment[batchid][indd]].detach().cpu().numpy())
        iter += 1
        # if it % cfg['log']['vis_step'] == 0:
        #     # visualize the first batch
        #     pc_util.write_ply(batch_anchor[indd].detach().cpu().numpy().reshape(-1,3),
        #                       os.path.join(cfg['log']['path'], 'visualization','ep{}_it{}_batch{}_svote.ply'.format(epoch, it, batchid)))
        #     pc_util.write_ply(surface_points[object_assignment[batchid][indd]].detach().cpu().numpy().reshape(-1, 3),
        #                       os.path.join(cfg['log']['path'], 'visualization',
        #                                    'ep{}_it{}_batch{}_gtsvote.ply'.format(epoch, it, batchid)))

    anchor_loss = avg_dist / (bs * shape_num)
    return anchor_loss


def compute_box_and_sem_cls_loss(est_data, gt_data, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        est_data, gt_data, meta_data: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    mean_size_arr = config.mean_size_arr

    object_assignment = est_data['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = est_data['center']
    gt_center = gt_data['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = gt_data['box_label_mask']
    objectness_label = est_data['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(gt_data['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_class_loss = criterion_heading_class(est_data['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(gt_data['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                   1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(
        torch.sum(est_data['heading_residuals_normalized'] * heading_label_one_hot,
                  -1) - heading_residual_normalized_label, delta=1.0)  # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss * objectness_label) / (
                torch.sum(objectness_label) + 1e-6)

    # Compute size loss
    size_class_label = torch.gather(gt_data['size_class_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    size_class_loss = criterion_size_class(est_data['size_scores'].transpose(2, 1), size_class_label)  # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    size_residual_label = torch.gather(gt_data['size_residual_label'], 1,
                                       object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(est_data['size_residuals_normalized'] * size_label_one_hot_tiled,
                                                   2)  # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(
        0)  # (1,1,num_size_cluster,3)
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)
    size_residual_normalized_loss = torch.mean(
        huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0),
        -1)  # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label) / (
                torch.sum(objectness_label) + 1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(gt_data['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    sem_cls_loss = criterion_sem_cls(est_data['sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    center_label = torch.gather(gt_center, 1, object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # selec
    gathered_gt = {}
    gathered_gt['center_label'] =  center_label
    gathered_gt['heading_class_label'] =  heading_class_label
    gathered_gt['heading_residual_label'] =  heading_residual_label
    gathered_gt['size_class_label'] =  size_class_label
    gathered_gt['size_residual_label'] =  size_residual_label
    return center_loss, heading_class_loss, heading_residual_normalized_loss\
        , size_class_loss, size_residual_normalized_loss, sem_cls_loss, gathered_gt #


def compute_quad_loss(est_data, gt_data):
    """ Compute 3D bounding box and semantic classification loss. """
    has_quad_ind = gt_data['use_quad'].bool()
    if est_data['quad_assignment'] == None:
        return torch.tensor(0),torch.tensor(0), torch.tensor(0),None

    quad_assignment = est_data['quad_assignment']
    quad_label = est_data['quad_label'].float()
    B = quad_assignment.shape[0]
    K = quad_assignment.shape[1]

    # Compute center loss
    pred_center = est_data['quad_center'][has_quad_ind, :, :]
    gt_center = gt_data['gt_quad_centers'][has_quad_ind, :, 0:3]
    quad_assignment_expand = quad_assignment.unsqueeze(2).repeat(1, 1, 3)
    assigned_gt_center = torch.gather(gt_center, 1, quad_assignment_expand)  # (B, K, 3) from (B, K2, 3)
    center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=1.0)  # (B,K)
    center_loss = torch.sum(center_loss * quad_label.unsqueeze(2)) / (torch.sum(quad_label) + 1e-6)

    # Compute normal vector loss
    pred_vector = est_data['normal_vector'][has_quad_ind, :, :]  # B,K,3
    gt_vector = torch.gather(gt_data['gt_normal_vectors'][has_quad_ind, :, :], 1, quad_assignment_expand)

    cos_similar = torch.cosine_similarity(pred_vector, gt_vector, dim=2)
    vector_loss = torch.ones((B, K), dtype=torch.float32).cuda() - cos_similar  # (B,K)
    vector_loss = torch.sum(vector_loss * quad_label) / (torch.sum(quad_label) + 1e-6)
    vector_loss = vector_loss

    # Compute size loss
    pred_size = est_data['quad_size'][has_quad_ind, :, :]
    gt_size = torch.gather(gt_data['gt_quad_sizes'][has_quad_ind, :, :], 1,
                           quad_assignment.unsqueeze(2).repeat(1, 1, 2))  # (B, K, 3) from (B, K2, 3)
    size_loss = smoothl1_loss(pred_size - gt_size, delta=1.0)
    size_loss = torch.sum((size_loss * quad_label.unsqueeze(2)) / (
            torch.sum(quad_label) + 1e-6))
    size_loss = size_loss

    gathered_gt = {}
    gathered_gt['quad_center_label'] =  assigned_gt_center
    gathered_gt['quad_vector_label'] =  gt_vector
    gathered_gt['quad_size_label'] =  gt_size

    return center_loss, vector_loss, size_loss, gathered_gt


def compute_quad_score_loss(est_data, gt_data):
    # Associate proposal and GT objects by point-to-point distances
    has_quad_ind = gt_data['use_quad'].bool()
    if len(gt_data['use_quad'][gt_data['use_quad'] > 0]) == 0:
        return torch.tensor(0), None, None, None, torch.tensor(0), torch.tensor(0)
    num_gt_quads = gt_data['num_gt_quads'][has_quad_ind]
    max_num_gt_quads = torch.max(num_gt_quads).item()

    #为了减少计算量，只保留有框的gt_data，但因为每个batch的gt box的数量不一样，选取最大的那个数（有些batch还是有都是0的情况）
    gt_center = gt_data['gt_quad_centers'][has_quad_ind, 0:max_num_gt_quads, 0:3]  # B, K2, 3
    gt_normal_vector = gt_data['gt_normal_vectors'][has_quad_ind, 0:max_num_gt_quads, 0:2]  # B, K2, 3
    aggregated_vote_xyz = est_data['aggregated_vote_quad_xyz'][has_quad_ind, :, :]
    B = aggregated_vote_xyz.shape[0]
    K = aggregated_vote_xyz.shape[1]
    dist1, quad_assignment, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: BxK, dist2: BxK2

    quad_label = torch.zeros((B, K), dtype=torch.long).cuda()
    # Set assignment

    gt_centers = torch.gather(gt_center, 1, quad_assignment.unsqueeze(-1).repeat(1, 1, 3))
    gt_normal_vector = torch.gather(gt_normal_vector, 1, quad_assignment.unsqueeze(-1).repeat(1, 1, 2))
    canonical_xyz = aggregated_vote_xyz - gt_centers
    canonical_xyz_ = canonical_xyz[:,:,:2].unsqueeze(2) #B,K, 1,2
    gt_normal_vector_ = gt_normal_vector.unsqueeze(3) #B,K, 2,1
    dist = torch.abs(torch.einsum('baij,bajk->baik', (canonical_xyz_, gt_normal_vector_)).squeeze(3).squeeze(2)) / torch.norm(gt_normal_vector[:,:,:2] , p=2, dim=2)

    inside_mask = dist < 0.2 #到墙面的距离< 0.2
    pos_mask = (dist1 < QUAD_NEAR_THRESHOLD) & inside_mask
    quad_label[pos_mask] = 1

    quad_scores = est_data['quad_scores'][has_quad_ind, :, :]
    criterion = nn.CrossEntropyLoss(torch.Tensor(QUAD_CLS_WEIGHTS).cuda(), reduction='none')
    quad_scores_loss = criterion(quad_scores.transpose(2, 1), quad_label)
    quad_scores_loss = quad_scores_loss.mean()

    obj_pred_val = torch.argmax(est_data['quad_scores'][has_quad_ind, :, :], 2)  # B,K
    quad_accuracy = torch.sum(
        torch.logical_and(obj_pred_val == quad_label.long(), obj_pred_val == 1)).float() / torch.sum(
        obj_pred_val).item()
    quad_recall = torch.sum(
        torch.logical_and(obj_pred_val == quad_label.long(), obj_pred_val == 1)).float() / torch.sum(quad_label).item()

    return quad_scores_loss, quad_label, None, quad_assignment, quad_accuracy, quad_recall


def compute_seg_loss(est_data,gt_data):
    instance_point_masks = gt_data['instance_point_masks']
    loss = 0
    for i in range(len(est_data['mask_scores'])):
        sa_mask_score = est_data['mask_scores'][i]
        if sa_mask_score == None:
            continue
        mask_indices =  est_data['mask_indices'][i]
        sa_mask_targets = pointnet2_utils.gather_operation(instance_point_masks.unsqueeze(1).float(), mask_indices.int() )#B,C,N, B,N
        sa_mask_targets = sa_mask_targets.squeeze(1).long()
        sample_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, 0.8]).cuda(), reduction='none')
        loss += sample_loss(sa_mask_score, sa_mask_targets).mean()
    return loss


@LOSSES.register_module
class DetectionLoss(BaseLoss):
    def __call__(self, est_data, gt_data, dataset_config):
        loss_dict = {}
        # Vote loss
        vote_loss = compute_vote_loss(est_data, gt_data)
        loss_dict['votes_loss'] = vote_loss

        
        # Obj loss
        objectness_loss, objectness_label, object_assignment, objectness_recall, objectness_accuracy = \
                compute_objectness_loss(est_data, gt_data,dataset_config) #objectness_mask
        # Box loss and sem cls loss
        est_data['object_assignment'] = object_assignment
        est_data['objectness_label'] = objectness_label


        center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss, gathered_gt = \
            compute_box_and_sem_cls_loss(est_data, gt_data, dataset_config)
        box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss

        loss_dict['box_loss'] = box_loss.item()
        loss_dict['center_loss'] = center_loss.item()
        loss_dict['heading_cls_loss'] = heading_cls_loss.item()
        loss_dict['heading_reg_loss'] = heading_reg_loss.item()
        loss_dict['size_cls_loss'] = size_cls_loss.item()
        loss_dict['size_reg_loss'] = size_reg_loss.item()
        loss_dict['objectness_recall'] =  objectness_recall
        loss_dict['objectness_accuracy'] =objectness_accuracy
        loss_dict['objectness_loss'] =objectness_loss.item()

        # Final loss function
        if self.cfg.config['model']['detection']['shape']['supervise'] and 'anchors' in est_data:
            anchor_loss = compute_anchor_loss(est_data, gt_data, self.cfg.config, flag='detection')
            loss_dict['anchor_loss'] = anchor_loss.item()
            loss = vote_loss + 0.5 * objectness_loss + box_loss \
                   + 0.1 * sem_cls_loss + self.cfg.config['model']['detection']['shape']['wd0'] * anchor_loss \
                   # + iou_loss
        else:
            loss = vote_loss + 0.5 * objectness_loss + box_loss \
                   + 0.1 * sem_cls_loss #+ seg_loss*0.3


        loss *= 10

        loss_dict['object_total'] = loss
        return loss_dict

@LOSSES.register_module
class QuadDetectionLoss(BaseLoss):
    def __call__(self, est_data, gt_data, dataset_config):
        # quadness loss
        loss_dict = {}

        quad_scores_loss, quad_label, quad_mask, quad_assignment, quad_accuracy, quad_recall \
            = compute_quad_score_loss(est_data, gt_data)
        loss_dict['quad_scores_loss'] = quad_scores_loss.item()
        loss_dict['quad_accuracy'] = quad_accuracy.item()
        loss_dict['quad_recall'] = quad_recall.item()

        est_data['quad_assignment'] = quad_assignment
        est_data['quad_label'] = quad_label
        est_data['quad_mask'] = quad_mask

        # quad loss
        quad_center_loss, normal_vector_loss, quad_size_loss, gathered_gt = compute_quad_loss(est_data, gt_data)
        loss_dict['quad_center_loss'] = quad_center_loss.item()
        loss_dict['normal_vector_loss'] =normal_vector_loss.item()
        loss_dict['quad_size_loss'] = quad_size_loss.item()

        # quad vote loss
        quad_vote_loss = compute_quad_vote_loss(est_data, gt_data)
        loss_dict['quad_vote_loss'] = quad_vote_loss.item()

        # has_quad_ind = gt_data['use_quad'].bool()
        # iou_loss = compute_quad_axis_aligned_iou_loss(has_quad_ind, est_data, gathered_gt)
        # loss_dict['quad_iou_loss'] = iou_loss.item()


        if self.cfg.config['model']['quad_detection']['shape']['supervise']:
            quad_surface_loss = compute_anchor_loss(est_data, gt_data, self.cfg.config, flag='quad')
            loss_dict['quad_surface_loss'] = quad_surface_loss.item()
            quad_loss = 0.5 * quad_scores_loss + quad_center_loss + normal_vector_loss + quad_size_loss + quad_vote_loss \
                        + self.cfg.config['model']['quad_detection']['shape']['wd0'] * quad_surface_loss \
                        # + iou_loss

        else:
            quad_loss = 0.1 * quad_scores_loss + quad_center_loss + normal_vector_loss + quad_size_loss + quad_vote_loss \
                        # + iou_loss


        quad_loss *= 10

        quad_loss = quad_loss.to(gt_data['point_clouds'].device)
        loss_dict['quad_total'] = quad_loss

        return loss_dict

@LOSSES.register_module
class ChamferDist(BaseLoss):
    def __call__(self, pointset1, pointset2):
        '''
        calculate the chamfer distance between two point sets.
        :param pointset1 (B x N x 3): torch.FloatTensor
        :param pointset2 (B x N x 3): torch.FloatTensor
        :return:
        '''
        dist1, dist2 = chamfer_func(pointset1, pointset2)[:2]
        loss = self.weight * ((torch.mean(dist1)) + (torch.mean(dist2)))
        return loss

def compute_objectness_loss_new(est_data, gt_data, dataset_config, point_recall_thresh=0.1, DIST_THRESH=0.05, return_loss = True):
    # 3NN distance search
    # filter by inside mask and ceter distance threshold
    # compare point iou
    # reweight semantic labels (soft label)
    aggregated_vote_xyz = est_data['aggregated_vote_xyz']
    anchor_sampled_xyzs = gt_data['anchor_sampled_xyzs'] #.cpu().numpy()
    # anchor_sampled_xyzs = np.expand_dims(est_data['anchor_sampled_pts_list'],0) #.cpu().numpy()
    instance_pts = gt_data['instance_pts']
    instance_pts_numbers = gt_data['instance_pts_numbers']

    batch_size, Nproposal, _ = aggregated_vote_xyz.shape
    device = aggregated_vote_xyz.device

    gt_centers = gt_data['center_label'][:, :, 0:3]
    gt_cls_labels = gt_data['sem_cls_label']
    nums_gt_objects = gt_data['num_gt_objects']
    parsed_gts = parse_groundtruths(gt_data, dataset_config)

    ##outputs:
    objectness_label = torch.zeros((batch_size, Nproposal), dtype=torch.long).cuda()
    object_assignments = torch.zeros((batch_size, Nproposal), dtype=torch.long).cuda()
    soft_labels = torch.zeros((batch_size, Nproposal,8)).cuda()

    for bid in range(batch_size):
        # batch_pc = gt_data['point_clouds'][bid]
        batch_num_gt_objects = nums_gt_objects[bid][0].item()
        if batch_num_gt_objects ==0:
            continue
        # batch_gt_cls_labels = gt_cls_labels[bid][:batch_num_gt_objects]
        if batch_num_gt_objects < 2:   K = 1
        elif batch_num_gt_objects < 3: K = 2
        else:                    K = 3
        batch_gt_center = gt_centers[bid, :batch_num_gt_objects, 0:3]
        batch_pred_center = aggregated_vote_xyz[bid]
        '''
         First, find the nearest K(3) bboxes as the candidate bboxes according to the distances between their centers, 
         and assign the predict bbox to these K(3) gt bbox accordingly (assignment_Ks)
        '''
        dists_Ks, assignment_Ks = knn_point(K, batch_gt_center.unsqueeze(0), batch_pred_center.unsqueeze(0),return_values=True)  # B, n_proposal, K;
        # Then, filter out the candidate bboxes whose centers are outside the current bbox
        gt_centers_ = torch.zeros((K, Nproposal, 3)).to(device)
        gt_headings_ = torch.zeros((K, Nproposal, 1)).to(device)
        gt_sizes_ = torch.zeros((K, Nproposal, 3)).to(device)
        for k in range(K):
            gt_centers_[k] = torch.gather(batch_gt_center, 0,#B,Ngt,3 --B,N,K --> B,N,K,3
                                          assignment_Ks[0, :, k].unsqueeze(-1).repeat(1, 3))  # B, n_proposal, K, 3
            #                                  Ngt,3                          n_proposal -> n_proposal, 3
            gt_headings_[k] = torch.gather(torch.tensor(parsed_gts['heading_angles'][bid]).to(device), 0,
                                           assignment_Ks[0, :, k].unsqueeze(-1))  # B, n_proposal, K, 3
            #                                  Ngt                          n_proposal
            gt_sizes_[k] = torch.gather(torch.tensor(parsed_gts['box_sizes'][bid]).to(device), 0,
                                        assignment_Ks[0, :, k].unsqueeze(-1).repeat(1, 3))  # B, n_proposal, K, 3
            #                                  Ngt,3                          n_proposal -> n_proposal, 3
        canonical_xyz = batch_pred_center.unsqueeze(0).repeat(K, 1, 1) - gt_centers_  # K, Nproposal, 3
        from net_utils.box_util import rotation_3d_in_axis
        canonical_xyz = rotation_3d_in_axis(
            canonical_xyz.unsqueeze(2),
            -gt_headings_.squeeze(-1), 2).squeeze(2)
        distance_front = gt_sizes_[:, :, 0] - canonical_xyz[:, :, 0]
        distance_left = gt_sizes_[:, :, 1]/2. - canonical_xyz[:, :, 1] #条件变严格
        distance_top = gt_sizes_[:, :, 2]/2. - canonical_xyz[:, :, 2]
        distance_back = gt_sizes_[:, :, 0] + canonical_xyz[:, :, 0]
        distance_right = gt_sizes_[:, :, 1]/2. + canonical_xyz[:, :, 1]
        distance_bottom = gt_sizes_[:, :, 2]/2. + canonical_xyz[:, :, 2]

        distance_targets = torch.cat(
            (distance_front.unsqueeze(-1),
             distance_left.unsqueeze(-1),
             distance_top.unsqueeze(-1),
             distance_back.unsqueeze(-1),
             distance_right.unsqueeze(-1),
             distance_bottom.unsqueeze(-1)),
            dim=-1
        )  # 8,256,6
        # Filter out the candidate bboxes whose centers are outside the current bbox OR center distances larger than threshold
        inside_mask = (distance_targets >= 0.).all(dim=-1)  # K, Nproposal
        pos_mask = (dists_Ks[0].t()< NEAR_THRESHOLD**2) & inside_mask# & gt_valid_mask#K, Nproposal
        # pos_mask = inside_mask  # & gt_valid_mask#K, Nproposal
        best, _ = pos_mask.max(0)  # Nproposal
        ids = torch.where(best)[0]  # N_valid
        '''
         Finally, calculate the recall of the rest bboxes
         the recall is obtained based on the discrepency between anchor-sampled points and the GT instance points (nearest distance)
         adopt soft objectness label
        '''
        for vid in ids:  # Nproposal
            sampled_pts = anchor_sampled_xyzs[bid][vid]
            best_recall = 0.
            if pos_mask[:, vid].sum() > 1:  #if more than one of the K gt bboxes meet the standards to be selectedd
                for kid in range(K):
                    if pos_mask[kid, vid]:
                        gtid = assignment_Ks[0, vid, kid]
                        tgt_gt_instance_pts = instance_pts[bid,gtid][:instance_pts_numbers[bid][0].item(),:].cpu().numpy()
                        gt2pred_dist_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree',
                                                           metric='l2').fit(sampled_pts)
                        gt2pred_dist = gt2pred_dist_nn.kneighbors(tgt_gt_instance_pts)[0]
                        intersect_gt2pred = len(np.where(gt2pred_dist.squeeze() < DIST_THRESH)[0])
                        # intersect = (np.in1d(pc_sampled_inds, valid_gt_shapenet_instance_inds)).sum()
                        # point_iou = intersect / (len(sampled_inds)+len(valid_gt_shapenet_instance_inds - intersect))
                        recall = intersect_gt2pred / len(tgt_gt_instance_pts)
                        if recall == 0:
                            continue  # 当recall=0时，kid无效
                        # pred2gt_dist_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree',
                        #                                    metric='l2').fit(tgt_gt_instance_pts)
                        # pred2gt_dist = pred2gt_dist_nn.kneighbors(sampled_pts)[0]
                        # intersect_pred2gt = len(np.where(pred2gt_dist.squeeze() < DIST_THRESH)[0])
                        # precision = intersect_pred2gt / len(sampled_pts)
                        # intersect_score = recall * 2 + precision  # recall更重要

                        cls_id = gt_cls_labels[bid, gtid].item()
                        if recall > soft_labels[bid, vid, cls_id]:
                            soft_labels[bid, vid, cls_id] = recall  # intersect_score
                            # write_ply(tgt_gt_instance_pts,"temp/tgt_gt_instance_pts_%s_recall-%s.ply"%(vid,recall))
                            # write_ply(sampled_pts,"temp/sampled_pts_%s_recall-%s.ply"%(vid,recall))

                        if recall > best_recall:
                            best_recall = recall
                            best_kid = kid
                            # write_ply(tgt_gt_instance_pts,"temp/tgt_gt_instance_pts_%s_recall-%s.ply"%(vid.item(),recall))
                            # write_ply(sampled_pts,"temp/sampled_pts_%s_recall-%s.ply"%(vid.item(),recall))
            else:
                best_kid = torch.where(pos_mask[:, vid] > 0)[0].item()
                gtid = assignment_Ks[0, vid, best_kid]
                tgt_gt_instance_pts = instance_pts[bid,gtid][:instance_pts_numbers[bid,gtid].item(),:].cpu().numpy()
                gt2pred_dist_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree',
                                                   metric='l2').fit(sampled_pts)
                gt2pred_dist = gt2pred_dist_nn.kneighbors(tgt_gt_instance_pts)[0]
                intersect_gt2pred = len(np.where(gt2pred_dist.squeeze() < DIST_THRESH)[0])
                best_recall = intersect_gt2pred / len(tgt_gt_instance_pts)

                # pred2gt_dist_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree',
                #                                    metric='l2').fit(tgt_gt_instance_pts)
                # pred2gt_dist = pred2gt_dist_nn.kneighbors(sampled_pts)[0]
                # intersect_pred2gt = len(np.where(pred2gt_dist.squeeze() < DIST_THRESH)[0])
                # precision = intersect_pred2gt / len(sampled_pts)
                # intersect_score = recall * 2 + precision  # recall更重要
                soft_labels[bid, vid, gt_cls_labels[bid, gtid].item()] = best_recall

            if best_recall > point_recall_thresh:
                objectness_label[bid,vid]=1
                object_assignments[bid,vid]= assignment_Ks[0, vid, best_kid]
                # write_ply(tgt_gt_instance_pts, "temp/tgt_gt_instance_pts_%s_recall-%s.ply" % (vid.item(), best_recall))
                # write_ply(sampled_pts, "temp/sampled_pts_%s_recall-%s.ply" % (vid.item(), best_recall))
            # cls_label = batch_gt_cls_labels[assignment_Ks[0, vid, best_kid]].item()

    #vis for verfication:
    if return_loss:
        objectness_scores = est_data['objectness_scores']
        objectness_loss = objectness_criterion(objectness_scores.transpose(2, 1), objectness_label)
        objectness_loss = objectness_loss.mean()
        soft_labels = soft_labels/(soft_labels.sum(-1)+1e-6).unsqueeze(-1)
        # compute accuracy and recall
        obj_logits = objectness_scores.clone().squeeze(2).detach().cpu().numpy()
        obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
        objectness_label_numpy = objectness_label.detach().cpu().numpy()
        num_obj_correct = (objectness_label_numpy  == (obj_prob > 0.5)).sum()
        accuracy = num_obj_correct / len(obj_prob.flatten())
        #outputs: scorse
        return objectness_loss, objectness_label, object_assignments, soft_labels, accuracy

    return objectness_label, object_assignments, soft_labels


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