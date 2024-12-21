import numpy as np
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils import pc_util
from net_utils.nms import nms_3d_faster
from net_utils.libs import flip_axis_to_camera,softmax, flip_axis_to_depth
from net_utils.box_util import get_3d_box
from net_utils.eval_det import eval_det_multiprocessing_wo_mesh, get_iou_obb
# from net_utils.libs import sigmoid

LENGTH = 0.1
MAX_NUM_QUAD = 32

DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.

def get_normal_size(vert1, vert2):
    dy = vert2[1]-vert1[1]
    dx = vert2[0]-vert1[0]
    #朝向不重要
    normal = np.array([-dy,dx,0])
    normal = normal/ max(np.linalg.norm(normal),1e-6)
    return normal, np.linalg.norm(vert2-vert1)

def contain_point( pointlist, point):
    for ind in range(len(pointlist)):
        p = pointlist[ind]
        if np.linalg.norm(np.array(p -  point)) <=0.4:
            return True, p, ind
    return False, None, None

def get_ceiling_and_floor(pred_corners):
    floors = []
    heights = []
    for quad_corner in pred_corners:
        for i in range(2, 4):
            contain, p, ind = contain_point(floors, quad_corner[i])
            if not contain:
                floors.append(quad_corner[i])
            else:
                del floors[ind]
                new_corner = (p + quad_corner[i]) / 2
                floors.append(new_corner)
            heights.append(quad_corner[3,2]-quad_corner[1,2])

    floors = adjust_pts_order(np.array(floors))
    height = np.max(np.abs(np.array(heights)))
    ceilings = floors.copy()
    ceilings[:,2] = ceilings[:,2] + height
    return ceilings, floors, height

def generate_quad_bbox(floor_cor, height):
    obbs = []
    for i in range(len(floor_cor)):
        vert1 = floor_cor[i]
        if i == len(floor_cor)-1:
            vert2 =  floor_cor[0]
        else:
            vert2 = floor_cor[i+1]
        pred_center = (vert1 + vert2)/2
        pred_center[2] += height/2
        normal, width = get_normal_size(vert1[0:2],vert2[0:2])

        cos_theta = torch.cosine_similarity(torch.tensor(normal), torch.tensor([0, 1, 0]),
                                            dim=0)
        heading_angle = torch.arccos(cos_theta)
        cos_theta1 = torch.cosine_similarity(torch.tensor(normal),
                                             torch.tensor([1, 0, 0]),
                                             dim=0)
        if cos_theta1 > 0:
            heading_angle = np.pi * 2 - heading_angle
        obb = np.zeros((7,))
        obb[0:3] = pred_center
        obb[3] = width+0.1
        obb[4] = 0.1
        obb[5] = height
        obb[6] = heading_angle
        obbs.append(obb)
    obbs = np.vstack(tuple(obbs))
    return obbs

def adjust_pts_order(pts_3ds):
    ''' sort rectangle points by counterclockwise '''
    cen_x, cen_y, _ = np.mean(pts_3ds, axis=0)
    #refer_line = np.array([10,0])
    d2s = []
    for i in range(len(pts_3ds)):
        o_x = pts_3ds[i][0] - cen_x
        o_y = pts_3ds[i][1] - cen_y
        atan2 = np.arctan2(o_y, o_x)
        if atan2 < 0:
            atan2 += np.pi * 2
        d2s.append([pts_3ds[i], atan2])
    d2s = sorted(d2s, key=lambda x:x[1])
    order_2ds = np.array([x[0] for x in d2s])
    return order_2ds

def get_verts(center,width,height,normal_vector):

    normal_vector = normal_vector/max(np.linalg.norm(normal_vector),1e-6)
    center = np.array(center)
    normal_vector = np.array(normal_vector)
    x1 = center[0] + width * normal_vector[1] /2
    x2 = center[0] - width * normal_vector[1] /2

    x=[x1,x2]

    y1 = center[1] - width * normal_vector[0] /2
    y2 = center[1] + width * normal_vector[0] /2

    y=[y1,y2]

    h1 = center[2] + height/2
    h2 = center[2] - height/2

    h=[h1,h2]

    corners = []

    for _ in h:
        corners.append([x1,y1,_])
        corners.append([x2,y2,_])

    return np.array(corners)

def get_quad_bbox(end_points, has_quad_ind):
    pred_center = end_points['quad_center'][has_quad_ind,:,:]  # B,num_proposal,3
    pred_size = end_points['quad_size'][has_quad_ind,:,:]
    normal_vector = end_points['normal_vector'][has_quad_ind,:,:]
    num_proposal = pred_center.shape[1]

    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())

    pred_corners = np.zeros((bsize, num_proposal, 4, 3))
    obbs = np.zeros((bsize, num_proposal, 7))
    for i in range(bsize):
        for j in range(num_proposal):
            cos_theta = torch.cosine_similarity(torch.tensor(normal_vector[i, j, :].detach().cpu().numpy()),
                                                torch.tensor([0, 1, 0]), dim=0)
            heading_angle = torch.arccos(cos_theta)
            cos_theta1 = torch.cosine_similarity(torch.tensor(normal_vector[i, j, :].detach().cpu().numpy()),
                                                 torch.tensor([1, 0, 0]), dim=0)
            if cos_theta1 > 0:
                heading_angle = np.pi * 2 - heading_angle
            width = pred_size[i, j, 0].detach().cpu().numpy()
            height = pred_size[i, j, 1].detach().cpu().numpy()
            box_size = np.array([width, LENGTH, height])
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

            pred_corners[i, j, :] = get_verts(pred_center[i, j, :].detach().cpu().numpy(), width, height,
                                              normal_vector[i, j, :].detach().cpu().numpy())
            obbs[i, j, :3] = pred_center[i, j, :].detach().cpu().numpy()
            obbs[i, j, 3] = width
            obbs[i, j, 4] = LENGTH
            obbs[i, j, 5] = height
            obbs[i, j, 6] = heading_angle

    return pred_corners_3d_upright_camera, obbs, pred_corners

def get_quad_bbox_for_visualize(end_points, parsed_predictions, batchid):
    quad_prob = parsed_predictions['pred_quad_obj_prob'][batchid]
    pred_mask =  parsed_predictions['pred_quad_mask']
    ind = np.logical_and(quad_prob > DUMP_CONF_THRESH, pred_mask[batchid, :] == 1)

    pred_center = end_points['quad_center'][:,ind,:].detach().cpu().numpy() # B,num_proposal,3
    pred_size = end_points['quad_size'][:,ind,:].detach().cpu().numpy()
    normal_vector = end_points['normal_vector'][:,ind,:].detach().cpu().numpy()
    num_proposal = pred_center.shape[1]

    obbs = []
    pred_corners = []
    for j in range(num_proposal):
        cos_theta = torch.cosine_similarity(torch.tensor(normal_vector[batchid, j, :]),
                                            torch.tensor([0, 1, 0]), dim=0)
        heading_angle = torch.arccos(cos_theta)
        cos_theta1 = torch.cosine_similarity(torch.tensor(normal_vector[batchid, j, :]),
                                             torch.tensor([1, 0, 0]), dim=0)
        if cos_theta1 > 0:
            heading_angle = np.pi * 2 - heading_angle

        obb = np.zeros((7,))
        obb[0:3] = pred_center[batchid, j]
        obb[3] = pred_size[batchid, j, 0]
        obb[4] = 0.1
        obb[5] = pred_size[batchid, j, 1]
        obb[6] = heading_angle
        obbs.append(obb)

        width = pred_size[batchid, j, 0]
        height = pred_size[batchid, j, 1]

        pred_corner = get_verts(pred_center[batchid, j, :], width, height,
                                      normal_vector[batchid, j, :])
        pred_corners.append(pred_corner)

    ceilings, floors, max_height = get_ceiling_and_floor(pred_corners) # 4 points, first two ceilings, last two floors
    processed_obbs = generate_quad_bbox(floors, max_height)
    obbs = np.vstack(tuple(obbs))  # (num_proposal, 7)
    ceilings = np.vstack(tuple(ceilings))
    floors  = np.vstack(tuple(floors))

    return obbs, processed_obbs, floors, ceilings, max_height

def get_gt_quad_bbox_for_visualize(gt_data, batchid):
    center_label = gt_data['gt_quad_centers'].detach().cpu().numpy()
    size_label = gt_data['gt_quad_sizes'].detach().cpu().numpy()
    vector_label = gt_data['gt_normal_vectors'].detach().cpu().numpy()
    num_gt_quads = gt_data['num_gt_quads'].detach().cpu().numpy()

    # for i in range(batch_size):
    # Dump GT bounding boxes
    obbs = []
    gt_corners = []
    for j in range(num_gt_quads[batchid, 0]):
        cos_theta = torch.cosine_similarity(torch.tensor(vector_label[batchid, j, :]), torch.tensor([0, 1, 0]),
                                            dim=0)
        heading_angle = torch.arccos(cos_theta)
        cos_theta1 = torch.cosine_similarity(torch.tensor(vector_label[batchid, j, :]), torch.tensor([1, 0, 0]),
                                             dim=0)
        if cos_theta1 > 0:
            heading_angle = np.pi * 2 - heading_angle
        obb = np.zeros((7,))
        obb[0:3] = center_label[batchid, j]
        obb[3] = size_label[batchid, j, 0]
        obb[4] = 0.1
        obb[5] = size_label[batchid, j, 1]
        obb[6] = heading_angle
        obbs.append(obb)

        width = size_label[batchid, j, 0]
        height = size_label[batchid, j, 1]
        gt_corner = get_verts(center_label[batchid, j, :], width, height,
                                vector_label[batchid, j, :])

        gt_corners.append(gt_corner)

    ceilings, floors, _ = get_ceiling_and_floor(gt_corners)  # 4 points, first two ceilings, last two floors
    obbs = np.vstack(tuple(obbs))  # (num_proposal, 7)
    ceilings = np.vstack(tuple(ceilings))
    floors  = np.vstack(tuple(floors ))

    return obbs, floors, ceilings

def parse_quad_predictions(parsed_predictions, eval_dict, end_points, has_quad_ind, cfg):
    '''
    Args:
        parsed_predictions:
        eval_dict:
        end_points:
        config_dict:
    Returns:
        parsed_predictions: pred_quad_camera (B,Nquads, 8,3); pred_quad_obj_prob: (B,Nquads)
        eval_dict:
            pred_quad_mask (B,Nquads);
            batch_pred_quad_map_cls: list B x Nquads of tuple ( 1, array(8,3), array(1,) )
            batch_pred_quad_corners_list  = batch_pred_corners_list

    '''
    config_dict = cfg.eval_config
    # dataset_config = cfg.dataset_config
    pred_center = end_points['quad_center'][has_quad_ind,:,:]  # B,num_proposal,3
    bsize = pred_center.shape[0]
    K = pred_center.shape[1]  # K==num_proposal

    pred_corners_3d_upright_camera, box_params, pred_corners = get_quad_bbox(end_points, has_quad_ind)
    nonempty_box_mask = np.ones((bsize, K))
    obj_logits = end_points['quad_scores'][has_quad_ind,:,:].clone().squeeze(2).detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)

    # ---------- NMS input: pred_with_prob in (B,K,7) -----------
    pred_mask = np.zeros((bsize, K))
    for i in range(bsize):
        boxes_3d_with_prob = np.zeros((K, 7))
        for j in range(K):
            boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
            boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
            boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
            boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
            boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
            boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
            boxes_3d_with_prob[j, 6] = obj_prob[i, j]
        nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
        pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                             config_dict['nms_iou'], config_dict['use_old_type_nms'])
        assert (len(pick) > 0)
        pred_mask[i, nonempty_box_inds[pick]] = 1
    # ---------- NMS output: pred_mask in (B,K) -----------


    batch_pred_corners_list = []
    batch_pred_map_cls = []

    for i in range(bsize):
        batch_pred_map_cls.append([(1, pred_corners_3d_upright_camera[i, j], obj_prob[i, j]) \
                                   for j in range(pred_center.shape[1]) if
                                   pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['quad_conf_thresh']])
        batch_pred_corners_list.append([pred_corners[i, j] \
                                        for j in range(pred_center.shape[1]) if
                                        pred_mask[i, j] == 1 and obj_prob[i, j] > 0.5])

    parsed_predictions['pred_quad_camera'] = pred_corners_3d_upright_camera
    parsed_predictions['pred_oriented_quad_bboxes'] = box_params
    parsed_predictions['pred_quad_obj_prob'] = obj_prob
    parsed_predictions['pred_quad_mask'] = pred_mask
    eval_dict['batch_pred_quad_map_cls'] = batch_pred_map_cls
    eval_dict['batch_pred_quad_corners_list'] = batch_pred_corners_list

    return parsed_predictions, eval_dict

def parse_quad_groundtruths(eval_dict, gt_data, quad_thickness=None):
    """ Parse groundtruth labels to OBB parameters.

    Args:
        gt_data: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    has_quad_ind = gt_data['use_quad'].bool()
    bsize = torch.sum(has_quad_ind).item()
    center_label = gt_data['gt_quad_centers'][has_quad_ind]
    size_label = gt_data['gt_quad_sizes'][has_quad_ind]
    vector_label = gt_data['gt_normal_vectors'][has_quad_ind]
    num_gt_quads = gt_data['num_gt_quads'][has_quad_ind]
    quad_label_mask = gt_data['target_quad_mask'][has_quad_ind]

    K2 = MAX_NUM_QUAD  # num_gt_quads  # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    box3d = np.zeros((bsize, K2, 8, 3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:, :, 0:3].detach().cpu().numpy())

    gt_corners = np.zeros((bsize, K2, 4, 3))
    heading_angles = np.zeros((bsize, K2, 1))
    quad_sizes = np.zeros((bsize, K2, 3))
    bs_obbs = []

    for i in range(bsize):
        K2 = num_gt_quads[i,0]
        obbs = np.zeros((K2, 7))
        for j in range(K2):
            if quad_label_mask[i, j] == 0: break;
            cos_theta = torch.cosine_similarity(torch.tensor(vector_label[i, j, :].detach().cpu().numpy()),
                                                torch.tensor([0, 1, 0]), dim=0)
            heading_angle = torch.arccos(cos_theta)
            cos_theta1 = torch.cosine_similarity(torch.tensor(vector_label[i, j, :].detach().cpu().numpy()),
                                                 torch.tensor([1, 0, 0]), dim=0)
            if cos_theta1 > 0:
                heading_angle = np.pi * 2 - heading_angle
            width = size_label[i, j, 0].detach().cpu().numpy()
            height = size_label[i, j, 1].detach().cpu().numpy()
            if quad_thickness is not None:
                thickness = quad_thickness
            else:
                thickness = LENGTH
            box_size = np.array([width, thickness, height])

            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i, j, :])
            gt_corners_3d_upright_camera[i, j] = corners_3d_upright_camera
            box3d[i, j] = flip_axis_to_depth(corners_3d_upright_camera)
            gt_corners[i, j, :] = get_verts(center_label[i, j, :].detach().cpu().numpy(), width, height,
                                            vector_label[i, j, :].detach().cpu().numpy())
            obbs[j,:3] = center_label[i, j, :].detach().cpu().numpy()
            obbs[j,3] = width
            obbs[j,4] = thickness
            obbs[j,5] = height
            obbs[j,6] = heading_angle
            heading_angles[i,j] = heading_angle
            quad_sizes[i,j] = [width, thickness, height]

        bs_obbs.append(obbs)

    batch_gt_map_cls = []
    batch_gt_corners_list = []
    batch_gt_horizontal_list = []

    for i in range(bsize):
        batch_gt_map_cls.append([(1, gt_corners_3d_upright_camera[i, j]) for j in
                                 range(gt_corners_3d_upright_camera.shape[1]) if (j < num_gt_quads[i, j])])
        batch_gt_corners_list.append(
            [gt_corners[i, j] for j in range(gt_corners.shape[1]) if (j < num_gt_quads[i, j])])

        batch_gt_horizontal_list.append(gt_data['horizontal_quads'][has_quad_ind])

    eval_dict['gt_quad_corners_3d_upright_camera'] = gt_corners_3d_upright_camera
    eval_dict['gt_quad_corners'] = gt_corners
    eval_dict['gt_quad_box_corners'] = box3d
    eval_dict['quad_label_mask'] = quad_label_mask
    eval_dict['heading_angles'] = heading_angles
    eval_dict['quad_sizes'] = quad_sizes

    eval_dict['has_quad_ind'] = has_quad_ind
    eval_dict['gt_quad_box_params'] = bs_obbs
    eval_dict['quad_label_mask'] = quad_label_mask

    eval_dict['batch_gt_quad_map_cls'] = batch_gt_map_cls
    eval_dict['batch_gt_quad_corners_list'] = batch_gt_corners_list
    eval_dict['batch_gt_quad_horizontal_list'] = batch_gt_horizontal_list

    return eval_dict

class QUADAPCalculator(object):
    ''' Calculating Average Precision '''
    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.distance_method = 'F1 score'
        self.reset()

    def step(self, batch_pred_map_cls, batch_gt_map_cls, batch_pred_corners_list, batch_gt_corners_list,
             batch_gt_horizontal_list, has_quad_ind):
        """ Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert (bsize == len(batch_gt_map_cls))

        for i in range(bsize):
            if has_quad_ind[i] == False:
                continue
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.pred_corners[self.scan_cnt] = batch_pred_corners_list[i]
            self.gt_corners[self.scan_cnt] = batch_gt_corners_list[i]
            self.horizontal_corners[self.scan_cnt] = batch_gt_horizontal_list[i]
            self.scan_cnt += 1

    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing_wo_mesh(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh,
                                                 get_iou_func=get_iou_obb)
        ret_dict = {}
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision' % (clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall' % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall' % (clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.pred_corners = {}
        self.gt_corners = {}
        self.horizontal_corners = {}
        self.scan_cnt = 0

    def same_point(self, pred, gt):
        distance = np.linalg.norm(np.array(pred - gt))
        return (distance <= 0.4)

    def compute_correctness(self, pred_corner, all_gt, is_embed=False):
        for gt in all_gt:
            correctness = True
            for i in range(0, 4):
                distance = np.linalg.norm(np.array(pred_corner[i] - gt[i]))
                if distance > 0.4:
                    correctness = False
            if correctness:
                return True
        return False

    def contain_point(self, pointlist, point):
        for p in pointlist:
            if self.same_point(p, point):
                return True, p
        return False, None

    def get_ceiling_and_floor(self, pred_corners):
        ceilings = []
        floors = []
        for quad_corner in pred_corners:
            for i in range(0, 2):
                contain, p = self.contain_point(ceilings, quad_corner[i])
                if not contain:
                    ceilings.append(quad_corner[i])
                else:
                    new_corner = (p + quad_corner[i]) / 2
                    ceilings.append(new_corner)

            for i in range(2, 4):
                contain, p = self.contain_point(floors, quad_corner[i])
                if not contain:
                    floors.append(quad_corner[i])
                else:
                    new_corner = (p + quad_corner[i]) / 2
                    floors.append(new_corner)

        return ceilings, floors

    def compute_F1(self, calculated=False):
        """
        find point radius < 0.4
        """
        tp = 0
        fn = 0
        fp = 0

        npos = 0
        for i in range(0, self.scan_cnt):
            npos += len(self.gt_corners[i])

        for i in range(0, self.scan_cnt):

            all_pred_corners = self.pred_corners[i]
            all_gt_corners = self.gt_corners[i]
            horizontal_quads = self.horizontal_corners[i]

            horizontal_quads = np.array(horizontal_quads.cpu())

            for pred_corner in all_pred_corners:
                if self.compute_correctness(pred_corner, all_gt_corners):
                    tp = tp + 1
                else:
                    fp = fp + 1

            if calculated == True:  # calculate horizontal quads
                ceilings, floors = self.get_ceiling_and_floor(all_pred_corners)
                if len(ceilings) == 4:
                    if self.compute_correctness(ceilings, horizontal_quads, True):
                        tp = tp + 1
                if len(floors) == 4:
                    if self.compute_correctness(floors, horizontal_quads):
                        tp = tp + 1

        print(tp, fp, fn, npos)
        p = tp / max((tp + fp), 1e-6)
        # r = tp/(tp+fn)
        r = tp / npos

        f1 = 2.0 * p * r / max((p + r), 1e-6)

        return f1

def visualize_quad(gt_data, our_data, cfg, dump_dir):
    end_points = our_data[0]
    eval_dict = our_data[2]

    obbs, processed_obbs, floor_corners, ceil_corners = get_quad_bbox_for_visualize(end_points, eval_dict, 0)

    if len(processed_obbs) > 0:
        pc_util.write_oriented_bbox(processed_obbs,  os.path.join(dump_dir, 'processed_quad.ply'))
        np.savez(os.path.join(dump_dir, 'processed_quad.npz'),obbs = processed_obbs)

    if len(obbs) > 0:
        pc_util.write_oriented_bbox(obbs,  os.path.join(dump_dir, 'pred_confident_nms_quad.ply'))
    #     np.savez(os.path.join(dump_dir, 'pred_confident_nms_quad.npz'),obbs = obbs)
    #
    #
    # if len(floor_corners)>0:
    #     np.savez(os.path.join(dump_dir, 'pred_quad_corners.npz'),
    #          floor_corners= floor_corners, ceil_corners = ceil_corners)
    # else:
    #     if len(floor_corners)<8:
    #         print('corners number < 4!\n')
    #
    # # LABELS
    # gt_obbs, gt_floor_corners, gt_ceil_corners = get_gt_quad_bbox_for_visualize(gt_data, 0)
    # if len(gt_obbs) > 0:
    #     obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
    #     pc_util.write_oriented_bbox(gt_obbs, os.path.join(dump_dir, 'gt_quad.ply'))
    #     np.savez(os.path.join(dump_dir, 'gt_quad.npz'),gt_obbs = gt_obbs)
    #     np.savez(os.path.join(dump_dir, 'gt_quad_corners.npz'),
    #              floor_corners= gt_floor_corners, ceil_corners = gt_ceil_corners)