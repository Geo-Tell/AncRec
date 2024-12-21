import numpy as np
from net_utils.eval_det import eval_det_multiprocessing_wo_mesh, eval_det_multiprocessing_w_mesh, get_iou_obb, \
    compute_mesh_iou, eval_det_multiprocessing_distance, eval_det_multiprocessing_wo_mesh2
import torch
from net_utils.box_util import get_3d_box, get_3d_box_cuda
from net_utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls, nms_3d_cls_conditioned
from net_utils.libs import softmax, flip_axis_to_camera, flip_axis_to_depth, extract_pc_in_box3d
import trimesh
from trimesh.exchange.binvox import voxelize_mesh
import os
from multiprocessing import Pool
from functools import partial
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path

transform_shapenet = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


class APCalculator(object):
    ''' Calculating Average Precision '''

    def __init__(self, ap_thresh=0.25, class2type_map=None, evaluate_mesh=False, distance_method='iou'):
        """
        Args:
            ap_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_thresh = ap_thresh
        self.class2type_map = class2type_map
        self.evaluate_mesh = evaluate_mesh
        self.distance_method = distance_method
        self.reset()

    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert (bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def step_pred(self, batch_pred_map_cls):
        bsize = len(batch_pred_map_cls)
        for i in range(bsize):
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def step_gt(self, batch_gt_map_cls):
        bsize = len(batch_gt_map_cls)
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.scan_cnt += 1

    def reset2(self):
        self.scan_cnt = 0

    def compute_metrics(self):
        if self.evaluate_mesh:
            return self.compute_metrics_w_mesh()
        else:
            return self.compute_metrics_wo_mesh()

    def compute_metrics_wo_mesh(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing_wo_mesh(self.pred_map_cls, self.gt_map_cls,
                                                         ovthresh=self.ap_thresh, get_iou_func=get_iou_obb)
        # rec, prec, ap = eval_det_multiprocessing_wo_mesh2(self.pred_map_cls, self.gt_map_cls,
        #                                                  ovthresh=self.ap_thresh, get_iou_func=get_iou_obb)
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

    def compute_metrics_w_mesh(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh) = eval_det_multiprocessing_w_mesh(self.pred_map_cls,
                                                                                          self.gt_map_cls,
                                                                                          ovthresh=self.ap_thresh,
                                                                                          get_iou_func=get_iou_obb,
                                                                                          get_iou_mesh=compute_mesh_iou)
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

        # for mesh
        for key in sorted(ap_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision_mesh' % (clsname)] = ap_mesh[key]
        ret_dict['mAP_mesh'] = np.mean(list(ap_mesh.values()))
        rec_list_mesh = []
        for key in sorted(ap_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall_mesh' % (clsname)] = rec_mesh[key][-1]
                rec_list_mesh.append(rec_mesh[key][-1])
            except:
                ret_dict['%s Recall_mesh' % (clsname)] = 0
                rec_list_mesh.append(0)
        ret_dict['AR_mesh'] = np.mean(rec_list_mesh)
        return ret_dict

    def compute_distance_metrics(self):

        (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh) = eval_det_multiprocessing_distance(self.pred_map_cls,
                                                                                                      self.gt_map_cls,
                                                                                                      thresh=self.ap_thresh,
                                                                                                     distance_method_name = self.distance_method)
        ret_dict = {}

        # for mesh
        for key in sorted(ap_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['AP_mesh        %s' % (clsname)] = ap_mesh[key]
            ret_dict['Precision_mesh %s' % (clsname)] = prec_mesh[key]
            ret_dict['Recall_mesh    %s' % (clsname)] = rec_mesh[key]
        ret_dict['mAP_mesh'] = np.mean(list(ap_mesh.values()))
        ret_dict['mean Precision_mesh'] = np.mean(list(prec_mesh.values()))
        ret_dict['mean Recall_mesh'] = np.mean(list(rec_mesh.values()))

        # for PQ
        for key in sorted(PQ_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['PQ_mesh %s' % (clsname)] = PQ_mesh[key]
            ret_dict['SQ_mesh %s' % (clsname)] = SQ_mesh[key]
            ret_dict['RQ_mesh %s' % (clsname)] = RQ_mesh[key]

        ret_dict['PQ_mesh'] = np.mean(list(PQ_mesh.values()))
        ret_dict['SQ_mesh'] = np.mean(list(SQ_mesh.values()))
        ret_dict['RQ_mesh'] = np.mean(list(RQ_mesh.values()))

        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0


def parse_predictions(parsed_predictions, eval_dict, est_data, gt_data, config_dict,
                      use_iou_for_nms = False, prefix='', return_batch_pred_map_cls=True):
    pred_center = est_data['center'+prefix]  # B,num_proposal,3
    pred_heading_class = torch.argmax(est_data['heading_scores'+prefix], -1)  # B,num_proposal
    heading_residuals = est_data['heading_residuals_normalized'+prefix] * (
                np.pi / config_dict['dataset_config'].num_heading_bin)  # Bxnum_proposalxnum_heading_bin
    pred_heading_residual = torch.gather(heading_residuals, 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(est_data['size_scores'+prefix], -1)  # B,num_proposal
    mean_size_arr =  torch.from_numpy(config_dict['dataset_config'].mean_size_arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mean_size_arr = mean_size_arr.to(pred_center.device)
    size_residuals = est_data['size_residuals_normalized'+prefix] * mean_size_arr  #.cuda()
    pred_size_residual = torch.gather(size_residuals, 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(est_data['sem_cls_scores'+prefix], -1)  # B,num_proposal
    sem_cls_probs = softmax(est_data['sem_cls_scores'+prefix].detach().cpu().numpy())  # B,num_proposal,10
    # pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal

    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    zmin = np.zeros((bsize, num_proposal))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())

    pred_headings = np.zeros(pred_heading_class.shape)
    pred_sizes = np.zeros((pred_heading_class.shape[0], pred_heading_class.shape[1], 3))

    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict['dataset_config'].class2angle( \
                pred_heading_class[i, j].detach().cpu().numpy(), pred_heading_residual[i, j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size( \
                int(pred_size_class[i, j].detach().cpu().numpy()), pred_size_residual[i, j].detach().cpu().numpy())

            pred_headings[i][j] = heading_angle
            pred_sizes[i][j] = box_size

            corners_3d_upright_camera = get_3d_box(box_size, -heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera
        if 'floor_height' in gt_data:
            box_corners_cam = pred_corners_3d_upright_camera[i]
            box_corners_depth = flip_axis_to_depth(box_corners_cam)
            centroid = (np.max(box_corners_depth, axis=1) + np.min(box_corners_depth, axis=1)) / 2.
            up_vector = box_corners_depth[:, 6] - box_corners_depth[:, 2]
            up_size = np.linalg.norm(up_vector, axis=1)
            zmin[i] = centroid[:,2]- up_size/2


    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = gt_data['point_clouds'].cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------
    above_floor_mask = np.ones((bsize, K))
    if 'floor_height' in gt_data:
        floor_height = gt_data['floor_height']
        above_floor_mask[floor_height.unsqueeze(1).repeat(1,K).cpu().numpy() - zmin > 0.1] = 0
        parsed_predictions['above_floor_mask'] = above_floor_mask

    obj_logits = est_data['objectness_scores'+prefix].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    nms_mask = np.zeros((bsize, K), dtype=np.uint8)
    if not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K), dtype=np.uint8)
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 2] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            nms_mask[i, pick] = 1
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1

        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K), dtype=np.uint8)
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
            nms_mask[i, pick] = 1
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1

        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        scores = obj_prob
        if use_iou_for_nms:
            iou_logits = torch.nn.Sigmoid()(est_data['iou_scores'])
            if iou_logits.shape[2] > 1:
                iou_logits = torch.gather(iou_logits, 2, pred_sem_cls.unsqueeze(-1))
            iou_logits = iou_logits.squeeze(-1).detach().cpu().numpy()
            # scores = iou_logits
            scores = scores * iou_logits
            parsed_predictions['iou_prob'] = iou_logits
        pred_mask = np.zeros((bsize, K), dtype=np.uint8)
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = scores[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[i, j].item()  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                         config_dict['nms_iou'], config_dict['use_old_type_nms'])
            # pick = nms_3d_cls_conditioned(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
            #                              config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
            nms_mask[i, pick] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    parsed_predictions['pred_sizes'] = pred_sizes
    parsed_predictions['pred_headings'] = pred_headings
    parsed_predictions['pred_centers'] = pred_center.detach().cpu().numpy() #B,N,3

    parsed_predictions['pred_corners_3d_upright_camera'] = pred_corners_3d_upright_camera
    parsed_predictions['sem_cls_probs'] = sem_cls_probs
    # parsed_predictions['obj_prob'] =  scores
    # parsed_predictions['iou_prob'] =  iou_logits
    parsed_predictions['obj_prob'] =  obj_prob
    parsed_predictions['pred_sem_cls'] = pred_sem_cls .cpu().numpy() #item()
    parsed_predictions['pred_mask'] = pred_mask
    parsed_predictions['nms_mask'] = nms_mask

    if 'valid_mask' in gt_data:
        pred_mask = pred_mask * gt_data['valid_mask'].cpu().numpy()

    if return_batch_pred_map_cls:
        bsize, N_proposals = pred_sem_cls.shape
        batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
        for i in range(bsize):
            if config_dict['per_class_proposal']:
                cur_list = []
                if config_dict['remove_under_floor']:
                    for ii in range(config_dict['dataset_config'].num_class):
                        cur_list += [
                            (ii, pred_corners_3d_upright_camera[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j]) \
                                     for j in range(N_proposals) if
                                     pred_mask[i, j] == 1 and above_floor_mask[i, j] == 1
                                     and obj_prob[i, j] > config_dict['conf_thresh']]
                else:
                    for ii in range(config_dict['dataset_config'].num_class):
                        cur_list += [
                            (ii, pred_corners_3d_upright_camera[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j]) \
                            for j in range(N_proposals) if
                            pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']]
                batch_pred_map_cls.append(cur_list)
            else:
                batch_pred_map_cls.append([(pred_sem_cls[i, j].item(),
                                            pred_corners_3d_upright_camera[i, j],
                                            obj_prob[i, j]) \
                                           for j in range(N_proposals) if
                                           pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])
        eval_dict['batch_pred_map_cls'] = batch_pred_map_cls

    eval_dict['pred_mask'] = pred_mask
    return parsed_predictions, eval_dict


def assembly_pred_map_cls(eval_dict, parsed_predictions, config_dict, mesh_outputs=None, voxel_size=0.047):
    pred_corners_3d_upright_camera = parsed_predictions['pred_corners_3d_upright_camera']
    sem_cls_probs = parsed_predictions['sem_cls_probs']
    obj_prob = parsed_predictions['obj_prob']
    # iou_prob = parsed_predictions['iou_prob']
    pred_mask = parsed_predictions['pred_mask']
    above_floor_mask = parsed_predictions['above_floor_mask']
    pred_sem_cls = parsed_predictions['pred_sem_cls']
    bsize, N_proposals = pred_sem_cls.shape
    if mesh_outputs is not None:
        assert bsize == 1
        meshes = mesh_outputs['meshes']
        proposal_ids = mesh_outputs['proposal_ids'].cpu().numpy()

    batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            if mesh_outputs is None:
                cur_list = []
                for ii in range(config_dict['dataset_config'].num_class):
                    cur_list += [(ii, pred_corners_3d_upright_camera[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j]) \
                                 for j in range(N_proposals) if
                                 pred_mask[i, j] == 1 and above_floor_mask[i,j]==1 and obj_prob[i, j] > config_dict['conf_thresh']]
                                 # and iou_prob[i, j] > 0.5]
            else:
                sample_idx = [(ii, j) for ii in range(config_dict['dataset_config'].num_class) for j in
                              range(N_proposals) if
                              pred_mask[i, j] == 1 and above_floor_mask[i,j]==1 and  obj_prob[i, j] > config_dict['conf_thresh']]
                              # and iou_prob[i, j] > 0.5]

                p = Pool(processes=16)
                cur_list = p.map(partial(batch_load_pred_data, proposal_ids=proposal_ids,
                                         batch_id=i, pred_corners=pred_corners_3d_upright_camera,
                                         sem_cls_probs=sem_cls_probs, obj_prob=obj_prob, meshes=meshes, voxel_size=voxel_size), sample_idx)
                p.close()
                p.join()

            batch_pred_map_cls.append(cur_list)
        else:
            if mesh_outputs is None:
                batch_pred_map_cls.append([(pred_sem_cls[i, j].item(),
                                            pred_corners_3d_upright_camera[i, j],
                                            obj_prob[i, j]) \
                                           for j in range(N_proposals) if
                                           pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])
            else:
                sample_idx = [j for j in range(N_proposals) if
                              pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']]
                p = Pool(processes=16)
                temp_list = p.map(partial(batch_load_pred_data_wo_cls, proposal_ids=proposal_ids, meshes=meshes,
                                          pred_corners=pred_corners_3d_upright_camera, batch_id=i,
                                          pred_sem_cls=pred_sem_cls.cpu().numpy(),
                                          obj_prob=obj_prob, voxel_size=voxel_size), sample_idx)
                p.close()
                p.join()

                batch_pred_map_cls.append(temp_list)

    eval_dict['batch_pred_map_cls'] = batch_pred_map_cls

    return eval_dict

def assembly_pred_map_cls2(parsed_predictions, conf_thresh=0.05):
    pred_corners_3d_upright_camera = parsed_predictions['pred_corners_3d_upright_camera']
    sem_cls_probs = parsed_predictions['sem_cls_probs']
    obj_prob = parsed_predictions['obj_prob']
    # pred_sem_cls = parsed_predictions['pred_sem_cls']
    N_proposals = len(obj_prob)
    batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    # for i in range(bsize):
    cur_list = []
    for ii in range(8):
        cur_list += [(ii, pred_corners_3d_upright_camera[j], sem_cls_probs[j, ii] * obj_prob[j]) \
                     for j in range(N_proposals) if
                     obj_prob[j] > conf_thresh]
    batch_pred_map_cls.append(cur_list)
    return batch_pred_map_cls


def parse_groundtruths(gt_data, dataset_config):
    """ Parse groundtruth labels to OBB parameters.

    Args:
        gt_data: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        dataset_config: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    center_label = gt_data['center_label']
    heading_class_label = gt_data['heading_class_label']
    heading_residual_label = gt_data['heading_residual_label']
    size_class_label = gt_data['size_class_label']
    size_residual_label = gt_data['size_residual_label']
    box_label_mask = gt_data['box_label_mask']
    sem_cls_label = gt_data['sem_cls_label']
    bsize = center_label.shape[0]

    K2 = center_label.shape[1]  # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    heading_angles = np.zeros((bsize, K2, 1))
    box_sizes = np.zeros((bsize, K2, 3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:, :, 0:3].detach().cpu().numpy())
    for i in range(bsize):
        for j in range(K2):
            if box_label_mask[i, j] == 0: continue
            heading_angles[i,j] = dataset_config.class2angle(heading_class_label[i, j].detach().cpu().numpy(),
                                                                      heading_residual_label[
                                                                          i, j].detach().cpu().numpy())
            box_sizes[i,j] = dataset_config.class2size(int(size_class_label[i, j].detach().cpu().numpy()),
                                                                size_residual_label[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_sizes[i,j], -heading_angles[i,j,0], gt_center_upright_camera[i, j, :])
            gt_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    return {'sem_cls_label': sem_cls_label,
            'gt_corners_3d_upright_camera': gt_corners_3d_upright_camera,
            'box_label_mask': box_label_mask,
            'heading_angles': heading_angles,
            'box_sizes': box_sizes,
            'box_centers': center_label}


def assembly_gt_map_cls(parsed_gts, mesh_outputs=None, voxel_size=0.047):
    sem_cls_label = parsed_gts['sem_cls_label']
    gt_corners_3d_upright_camera = parsed_gts['gt_corners_3d_upright_camera']
    box_label_mask = parsed_gts['box_label_mask']
    # scan_names = parsed_gts['scan_name']
    bsize = sem_cls_label.shape[0]
    MAX_OBJs = gt_corners_3d_upright_camera.shape[1]

    if mesh_outputs is not None:
        assert bsize == 1
        shapenet_catids = mesh_outputs['shapenet_catids'][0]
        shapenet_ids = mesh_outputs['shapenet_ids'][0]
        meshes = [
            trimesh.load(os.path.join(ShapeNetv2_Watertight_Scaled_Simplified_path, shapenet_catid, shapenet_id + '.off'),
                         process=False) for shapenet_catid, shapenet_id in zip(shapenet_catids, shapenet_ids)]

    batch_gt_map_cls = []
    for i in range(bsize):
        if mesh_outputs is None:
            batch_gt_map_cls.append([(sem_cls_label[i, j].item(), gt_corners_3d_upright_camera[i, j]) for j in
                                     range(MAX_OBJs) if box_label_mask[i, j] == 1]) #scan_names[i],
        else:
            sample_idx = [j for j in range(MAX_OBJs) if box_label_mask[i, j] == 1]

            p = Pool(processes=16)
            temp_list = p.map(partial(batch_load_gt_data, meshes=meshes, gt_corners=gt_corners_3d_upright_camera,
                                      sem_cls_label=sem_cls_label.cpu().numpy(), batch_id=i, voxel_size=voxel_size), sample_idx)
            p.close()
            p.join()
            batch_gt_map_cls.append(temp_list)

    return batch_gt_map_cls

'''evaluation of .pkl data: no batch size'''
def assembly_gt_map_cls2(parsed_gts):
    sem_cls_label = parsed_gts['sem_cls_label']
    gt_corners_3d_upright_camera = parsed_gts['gt_corners_3d_upright_camera']
    num_targets = len(gt_corners_3d_upright_camera)
    batch_gt_map_cls = []
    batch_gt_map_cls.append([(sem_cls_label[j].item(), gt_corners_3d_upright_camera[j]) for j in
                                     range(num_targets)]) #scan_names[i],
    return batch_gt_map_cls


def fit_shapenet_obj_to_votenet_box(points, box_corners):
    '''
    Fit points from shapenet objects to box corners produced from votenet.
    '''
    # recover box corners to 7-d coordinates
    corners_3d_depth = flip_axis_to_depth(box_corners)
    center = (np.max(corners_3d_depth, axis=0) + np.min(corners_3d_depth, axis=0)) / 2.
    forward_vector = corners_3d_depth[1] - corners_3d_depth[2]
    left_vector = corners_3d_depth[0] - corners_3d_depth[1]
    up_vector = corners_3d_depth[6] - corners_3d_depth[2]
    orientation = np.arctan2(forward_vector[1], forward_vector[0])
    sizes = np.linalg.norm([forward_vector, left_vector, up_vector], axis=1)

    # transform obj points to boxes
    obj_points = points - (points.max(0) + points.min(0)) / 2.
    obj_points = obj_points.dot(transform_shapenet.T)
    obj_points = obj_points.dot(np.diag(1 / (obj_points.max(0) - obj_points.min(0)))).dot(np.diag(sizes))

    axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0],
                               [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    obj_points = obj_points.dot(axis_rectified) + center

    return obj_points


def batch_load_pred_data(idx, proposal_ids, batch_id, pred_corners, sem_cls_probs, obj_prob, meshes, voxel_size):
    ii, j = idx

    mesh_data = meshes[list(proposal_ids[batch_id, :, 0]).index(j)]
    obj_points = mesh_data.vertices

    obj_points = fit_shapenet_obj_to_votenet_box(obj_points, pred_corners[batch_id, j])
    mesh_data.vertices = obj_points

    dimension = int(max((obj_points.max(0) - obj_points.min(0))) / voxel_size)
    dimension = max(dimension, 2)
    from pyvirtualdisplay import Display
    with Display(size=(100, 60)) as disp:  # backend="xvfb"
        # internal voxels
        voxel_data_internal = voxelize_mesh(mesh_data, dimension=dimension, wireframe=True, dilated_carving=True)
        # surface voxels
        voxel_data_surface = voxelize_mesh(mesh_data, exact=True, dimension=dimension)

    return (ii, pred_corners[batch_id, j], sem_cls_probs[batch_id, j, ii] * obj_prob[batch_id, j],
            (voxel_data_internal, voxel_data_surface), mesh_data)


def batch_load_pred_data_wo_cls(j, proposal_ids, meshes, pred_corners, batch_id, pred_sem_cls, obj_prob, voxel_size):
    mesh_data = meshes[list(proposal_ids[batch_id, :, 0]).index(j)]
    obj_points = mesh_data.vertices
    obj_points = fit_shapenet_obj_to_votenet_box(obj_points, pred_corners[batch_id, j])
    mesh_data.vertices = obj_points

    dimension = int(max((obj_points.max(0) - obj_points.min(0))) / voxel_size)
    dimension = max(dimension, 2)
    from pyvirtualdisplay import Display
    with Display(size=(100, 60)) as disp:  # backend="xvfb"
        # internal voxels
        voxel_data_internal = voxelize_mesh(mesh_data, dimension=dimension, wireframe=True, dilated_carving=True)
        # surface voxels
        voxel_data_surface = voxelize_mesh(mesh_data, exact=True, dimension=dimension)

    return (pred_sem_cls[batch_id, j], pred_corners[batch_id, j], obj_prob[batch_id, j],
            (voxel_data_internal, voxel_data_surface), mesh_data)


def batch_load_gt_data(j, meshes, gt_corners, sem_cls_label, batch_id, voxel_size):
    mesh_data = meshes[j]
    obj_points = mesh_data.vertices
    obj_points = fit_shapenet_obj_to_votenet_box(obj_points, gt_corners[batch_id, j])
    mesh_data.vertices = obj_points

    dimension = int(max((obj_points.max(0) - obj_points.min(0))) / voxel_size)
    dimension = max(dimension, 2)
    from pyvirtualdisplay import Display
    with Display(size=(100, 60)) as disp:  # backend="xvfb"
        # internal voxels
        voxel_data_internal = voxelize_mesh(mesh_data, dimension=dimension, wireframe=True, dilated_carving=True)
        # surface voxels
        voxel_data_surface = voxelize_mesh(mesh_data, exact=True, dimension=dimension)
    return (sem_cls_label[batch_id, j], gt_corners[batch_id, j], (voxel_data_internal, voxel_data_surface), mesh_data)

'''input: decoded scores; output:pred_corners_3d_upright_camera'''
def get_bbox(dataset_config, pred_center, size_cls_scores, size_residual_normalized, head_cls_scores, head_residual_normalized):
    batch_size = pred_center.shape[0]
    num_proposal = pred_center.shape[1]
    pred_heading_class = torch.argmax(head_cls_scores, -1)  # B,num_proposal
    heading_residuals = head_residual_normalized * (
                np.pi / dataset_config.num_heading_bin)  # Bxnum_proposalxnum_heading_bin
    pred_heading_residual = torch.gather(heading_residuals, 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(size_cls_scores, -1)  # B,num_proposal
    size_residuals = size_residual_normalized * torch.from_numpy(
        dataset_config.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
    pred_size_residual = torch.gather(size_residuals, 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)


    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.

    pred_corners_3d_upright_camera = np.zeros((batch_size, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(batch_size):
        for j in range(num_proposal):
            heading_angle = dataset_config.class2angle( \
                pred_heading_class[i, j].detach().cpu().numpy(), pred_heading_residual[i, j].detach().cpu().numpy())
            box_size = dataset_config.class2size( \
                int(pred_size_class[i, j].detach().cpu().numpy()), pred_size_residual[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, -heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    return pred_corners_3d_upright_camera

'''input: decoded scores; output:vis bbox parameters'''
def get_pred_bbox(est_data, dataset_config):
    pred_center = est_data['center']
    size_cls_scores = est_data['size_scores']
    size_residual_normalized = est_data['size_residuals_normalized']
    head_cls_scores = est_data['heading_scores']
    head_residual_normalized = est_data['heading_residuals_normalized']

    pred_heading_class = torch.argmax(head_cls_scores, -1)  # B,num_proposal
    heading_residuals = head_residual_normalized * (
            np.pi / dataset_config.num_heading_bin)  # Bxnum_proposalxnum_heading_bin
    pred_heading_residual = torch.gather(heading_residuals, 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(size_cls_scores, -1)  # B,num_proposal
    size_residuals = size_residual_normalized * torch.from_numpy(
        dataset_config.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
    pred_size_residual = torch.gather(size_residuals, 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)

    pred_angles = dataset_config.class2angle_cuda(pred_heading_class, pred_heading_residual)
    pred_sizes = dataset_config.class2size_cuda(pred_size_class, pred_size_residual)
    pred_corners = get_3d_box_cuda(pred_sizes, pred_angles, pred_center)

    ret = {}
    ret['box_centers'] = pred_center
    ret['box_sizes'] = pred_sizes
    ret['heading_angles'] = pred_angles
    ret['pred_corners'] =  pred_corners
    return ret

def get_pred_bbox_numpy(est_data, cfg, nms=False, gt_data=None):
    pred_center = est_data['center']
    dataset_config = cfg.dataset_config
    bsize, num_proposal, _ = pred_center.shape
    size_cls_scores = torch.tensor(est_data['size_scores'])
    size_residual_normalized = torch.tensor(est_data['size_residuals_normalized'])
    head_cls_scores = torch.tensor(est_data['heading_scores'])
    head_residual_normalized = torch.tensor(est_data['heading_residuals_normalized'])

    pred_heading_class = torch.argmax(head_cls_scores, -1)  # B,num_proposal
    heading_residuals = head_residual_normalized * (
            np.pi / dataset_config.num_heading_bin)  # Bxnum_proposalxnum_heading_bin
    pred_heading_residual = torch.gather(heading_residuals, 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(size_cls_scores, -1)  # B,num_proposal
    size_residuals = size_residual_normalized * torch.from_numpy(
        dataset_config.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
    pred_size_residual = torch.gather(size_residuals, 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_size_residual.squeeze_(2)
    pred_sem_cls = np.argmax(est_data['sem_cls_scores'], -1)  # B,num_proposal
    sem_cls_probs = softmax(est_data['sem_cls_scores'])  # B,num_proposal,10


    pred_angles = dataset_config.class2angle(pred_heading_class.numpy(), pred_heading_residual.numpy())
    pred_sizes = dataset_config.class2size(pred_size_class.numpy(), pred_size_residual.numpy())
    pred_corners = get_3d_box(pred_sizes, pred_angles, pred_center)

    parsed_predictions = {}
    parsed_predictions['box_centers'] = pred_center
    parsed_predictions['box_sizes'] = pred_sizes
    parsed_predictions['heading_angles'] = pred_angles
    parsed_predictions['pred_corners'] =  pred_corners
    parsed_predictions['pred_sem_cls'] = pred_sem_cls
    parsed_predictions['sem_cls_probs'] = sem_cls_probs
    obj_logits = est_data['objectness_scores']
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    parsed_predictions['obj_prob'] =  obj_prob


    if nms:
        nms_mask = np.zeros((bsize, num_proposal), dtype=np.uint8)
        nonempty_box_mask = np.ones((bsize, num_proposal))
        config_dict = cfg.config
        if config_dict['remove_empty_box'] and gt_data is not None:
            batch_pc = gt_data['point_clouds']
            for i in range(bsize):
                pc = batch_pc[i, :, :]  # (N,3)
                for j in range(num_proposal):
                    box3d = pred_corners [i, j, :, :]  # (8,3)
                    box3d = flip_axis_to_depth(box3d)
                    pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                    if len(pc_in_box) < 5:
                        nonempty_box_mask[i, j] = 0
        parsed_predictions['nonempty_box_mask'] = nonempty_box_mask
            # -------------------------------------
        above_floor_mask = np.ones((bsize, num_proposal))
        if 'floor_height' in gt_data:
            zmin = np.zeros((bsize, num_proposal))
            for bid in range(bsize):
                box_corners_cam = pred_corners[i]
                box_corners_depth = flip_axis_to_depth(box_corners_cam)
                centroid = (np.max(box_corners_depth, axis=1) + np.min(box_corners_depth, axis=1)) / 2.
                up_vector = box_corners_depth[:, 6] - box_corners_depth[:, 2]
                up_size = np.linalg.norm(up_vector, axis=1)
                zmin[i] = centroid[:, 2] - up_size / 2
            floor_height = gt_data['floor_height']
            above_floor_mask[floor_height.unsqueeze(1).repeat(1, num_proposal).cpu().numpy() - zmin > 0.1] = 0
            parsed_predictions['above_floor_mask'] = above_floor_mask

        scores = obj_prob

        pred_mask = np.zeros((bsize, num_proposal), dtype=np.uint8)
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((num_proposal, 8))
            for j in range(num_proposal):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = scores[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[i, j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            # pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
            #                              config_dict['nms_iou'], config_dict['use_old_type_nms'])
            pick = nms_3d_cls_conditioned(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                         config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
            nms_mask[i, pick] = 1

        parsed_predictions['pred_mask'] = pred_mask
        parsed_predictions['nms_mask'] = nms_mask

    return parsed_predictions


def get_gt_bbox(gt_data, dataset_config):
    """ Parse groundtruth labels to OBB parameters.

    Args:
        gt_data: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        dataset_config: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    box_label_mask = gt_data['box_label_mask']
    gt_box_num = int(torch.max(torch.sum(gt_data['box_label_mask'],dim=-1)).item())
    gt_centers = gt_data['center_label'][:,:gt_box_num].contiguous()
    heading_class_label = gt_data['heading_class_label'][:,:gt_box_num].contiguous()
    heading_residual_label = gt_data['heading_residual_label'][:,:gt_box_num].contiguous()
    size_class_label = gt_data['size_class_label'][:,:gt_box_num].contiguous()
    size_residual_label = gt_data['size_residual_label'][:,:gt_box_num].contiguous()
    sem_cls_label = gt_data['sem_cls_label'][:,:gt_box_num].contiguous()

    gt_angles = dataset_config.class2angle_cuda(heading_class_label,heading_residual_label)
    gt_sizes = dataset_config.class2size_cuda(size_class_label,size_residual_label)
    gt_corners = get_3d_box_cuda(gt_sizes, gt_angles, gt_centers)


    return {'sem_cls_label': sem_cls_label,
            'gt_corners': gt_corners,
            'box_label_mask': box_label_mask,
            'heading_angles': gt_angles,
            'box_sizes':  gt_sizes,
            'box_centers': gt_centers}

def assembly_refineed_pred_cls(data, est_data, config_dict, use_original_predictions = False, use_rgb_predictions = False):
    batch_pred_map_cls = []
    mask = data['valid_mask'].bool()
    pred_corners_3d_upright_camera = data['pred_corners_3d_upright_camera'][mask].cpu().numpy()
    if use_original_predictions:
        sem_cls_probs = data['pred_sem_cls_scores']
        obj_scores = data['pred_obj_scores'] .cpu().numpy()
    elif use_rgb_predictions and 'sem_cls_scores_rgb' in est_data:
        sem_cls_probs = est_data['sem_cls_scores_rgb'].detach()
        obj_scores = est_data['obj_scores_rgb'].cpu().numpy()
    else:
        sem_cls_probs = est_data['sem_cls_scores'].detach() #_final
        obj_scores = est_data['obj_scores'].cpu().numpy() #_final

    # obj_prob = softmax(obj_logits)[:, 1]
    # pred_sem_cls = torch.argmax(sem_cls_probs, -1)
    sem_cls_prob = softmax(sem_cls_probs.cpu().numpy())
    obj_prob = softmax(obj_scores)[:, 1] #obj_prob = softmax(obj_logits)[:, 1]
    batch_size = pred_corners_3d_upright_camera.shape[0]
    scan_names = data['scan_name']
    for bid in range(batch_size):
        # if config_dict['per_class_proposal']:
        for ii in range(config_dict['dataset_config'].num_class):
            if obj_prob[bid] > config_dict['conf_thresh']:
                batch_pred_map_cls.append(
                    (scan_names[bid], ii, pred_corners_3d_upright_camera[bid], sem_cls_prob[bid, ii] * obj_prob[bid])
                )
    batch_pred_map_cls = [batch_pred_map_cls]
    return batch_pred_map_cls

def assembly_gt_cls_new(data):
    valid_mask = data['valid_mask'].bool()
    sem_cls_label = data['gt_sem_cls_labels'][valid_mask].cpu().numpy()
    gt_corners_3d_upright_camera =  data['gt_corners_3d_upright_camera'][valid_mask].cpu().numpy()
    bsize = sem_cls_label.shape[0]
    scan_names = data['scan_name']
    gt_ids = data['gt_id'].cpu().numpy()
    objectness_labels = data['objectness_label']
    batch_gt_map_cls = []
    for i in range(bsize):
        if objectness_labels[i]:
            batch_gt_map_cls.append((scan_names[i], gt_ids[i], sem_cls_label[i], gt_corners_3d_upright_camera[i]))

    return [batch_gt_map_cls]
