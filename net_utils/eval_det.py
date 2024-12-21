from multiprocessing import Pool
from net_utils.box_util import box3d_iou
import numpy as np
from net_utils.metric_util import calc_iou # axis-aligned 3D box IoU
import trimesh
from sklearn.neighbors import NearestNeighbors
# from lfd import LightFieldDistance

def calc_lfd(pred, gt):
    from pyvirtualdisplay import Display
    with Display(size=(100, 60)) as disp:  # backend="xvfb"
        dist = LightFieldDistance().get_distance(pred.vertices, pred.faces, gt.vertices, gt.faces)
    return dist

def compute_mesh_iou(voxel1, voxel2):
    voxel1_internal, voxel1_surface = voxel1
    voxel2_internal, voxel2_surface = voxel2

    if voxel1_surface.filled_count ==0 or voxel2_surface.filled_count == 0:
        return 0.

    # (Note: internal voxels would be empty)
    if voxel1_internal.filled_count > 0 and voxel2_internal.filled_count > 0:
        v1_internal_points = voxel1_internal.points
        # v1 surface points that are not belong to internal.
        v1_surface_points = voxel1_surface.points[voxel1_internal.is_filled(voxel1_surface.points) == False]
        v1_points = np.vstack([v1_internal_points, v1_surface_points])

        v2_internal_points = voxel2_internal.points
        # v2 surface points that are not belong to internal.
        v2_surface_points = voxel2_surface.points[voxel2_internal.is_filled(voxel2_surface.points) == False]
        v2_points = np.vstack([v2_internal_points, v2_surface_points])

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points) + voxel2_internal.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points) + voxel1_internal.is_filled(v2_points))

    elif voxel1_internal.filled_count == 0 and voxel2_internal.filled_count > 0:
        v1_points = voxel1_surface.points

        v2_internal_points = voxel2_internal.points
        # v2 surface points that are not belong to internal.
        v2_surface_points = voxel2_surface.points[voxel2_internal.is_filled(voxel2_surface.points) == False]
        v2_points = np.vstack([v2_internal_points, v2_surface_points])

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points) + voxel2_internal.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points))

    elif voxel1_internal.filled_count > 0 and voxel2_internal.filled_count == 0:
        v2_points = voxel2_surface.points

        v1_internal_points = voxel1_internal.points
        # v1 surface points that are not belong to internal.
        v1_surface_points = voxel1_surface.points[voxel1_internal.is_filled(voxel1_surface.points) == False]
        v1_points = np.vstack([v1_internal_points, v1_surface_points])

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points) + voxel1_internal.is_filled(v2_points))
    else:
        v1_points = voxel1_surface.points
        v2_points = voxel2_surface.points

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points))

    if v1_in_v2 == 0 or v2_in_v1 == 0:
        return 0.

    alpha1 = v1_in_v2 / v1_points.shape[0]
    alpha2 = v2_in_v1 / v2_points.shape[0]

    return (alpha1 * alpha2) / (alpha1 + alpha2 - alpha1 * alpha2)


def get_iou_obb(bb1,bb2):
    iou3d, iou2d = box3d_iou(bb1,bb2)
    return iou3d

def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)

def chamfer_distance(x, y, metric='l2', direction='bi'):
    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = (np.mean(min_y_to_x) + np.mean(min_x_to_y)) / 2  # modified to keep scale
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist

def sample_and_calc_chamfer(pred, gt):
    # pred, gt: trimesh
    pred_points, _ = trimesh.sample.sample_surface(pred, 4096)
    gt_points, _ = trimesh.sample.sample_surface(gt, 4096)
    x = chamfer_distance(pred_points, gt_points)
    return x

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_iou(bb1, bb2):
    """ Compute IoU of two bounding boxes.
        ** Define your bod IoU function HERE **
    """
    #pass
    iou3d = calc_iou(bb1, bb2)
    return iou3d

def eval_det_cls_w_mesh(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou, get_iou_mesh=compute_mesh_iou):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {} # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array([item[0] for item in gt[img_id]])
        mesh = [item[1] for item in gt[img_id]]
        det = [False] * len(bbox)
        det_mesh = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det, 'mesh':mesh, 'det_mesh': det_mesh}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': [], 'mesh':[], 'det_mesh': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    meshes = []
    for img_id in pred.keys():
        for box,score,mesh in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
            meshes.append(mesh)
    confidence = np.array(confidence)
    BB = np.array(BB) # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    meshes = [meshes[x] for x in sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    tp_mesh = np.zeros(nd)
    fp_mesh = np.zeros(nd)
    for d in range(nd):
        if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d,...].astype(float)
        mesh_pred = meshes[d]

        ovmax = -np.inf
        ovmax_mesh = -np.inf
        BBGT = R['bbox'].astype(float)
        MESH_GT = R['mesh']

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

                iou_mesh = get_iou_main(get_iou_mesh, (mesh_pred, MESH_GT[j]))
                if iou_mesh > ovmax_mesh:
                    ovmax_mesh = iou_mesh
                    jmax_mesh = j

            # jmax_mesh = jmax
            # ovmax_mesh = get_iou_main(get_iou_mesh, (mesh_pred, MESH_GT[jmax_mesh]))

        #print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

        #print d, ovmax for mesh
        if ovmax_mesh > ovthresh:
            if not R['det_mesh'][jmax_mesh]:
                tp_mesh[d] = 1.
                R['det_mesh'][jmax_mesh] = 1
            else:
                fp_mesh[d] = 1.
        else:
            fp_mesh[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    #print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # for mesh
    # compute precision recall
    fp_mesh = np.cumsum(fp_mesh)
    tp_mesh = np.cumsum(tp_mesh)
    rec_mesh = tp_mesh / float(npos)
    #print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec_mesh = tp_mesh / np.maximum(tp_mesh + fp_mesh, np.finfo(np.float64).eps)
    ap_mesh = voc_ap(rec_mesh, prec_mesh, use_07_metric)

    return (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh)

def eval_det_cls_wo_mesh(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {} # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box,score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB) # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        #if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d,...].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        #print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    #print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def eval_det_cls_wrapper_w_mesh(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func, get_iou_mesh = arguments
    (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh) = eval_det_cls_w_mesh(pred, gt, ovthresh, use_07_metric, get_iou_func, get_iou_mesh)
    return (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh)

def eval_det_cls_wrapper_wo_mesh(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    rec, prec, ap = eval_det_cls_wo_mesh(pred, gt, ovthresh, use_07_metric, get_iou_func)
    return (rec, prec, ap)

def eval_det_multiprocessing_w_mesh(pred_all, gt_all, ovthresh=0.25, use_07_metric=True, get_iou_func=get_iou, get_iou_mesh=compute_mesh_iou):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score, voxel, _ in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score, voxel))
    for img_id in gt_all.keys():
        for classname, bbox, voxel, _ in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append((bbox, voxel))

    rec = {}
    prec = {}
    ap = {}
    rec_mesh = {}
    prec_mesh = {}
    ap_mesh = {}

    try:
        p = Pool(processes=8)
        ret_values = p.map(eval_det_cls_wrapper_w_mesh,
                           [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func, get_iou_mesh) for classname in
                            gt.keys() if classname in pred])
        p.close()
        p.join()
    except:
        ret_values = []
        for classname in gt.keys():
            if classname not in pred:
                continue
            ret_value = eval_det_cls_wrapper_w_mesh((pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func, get_iou_mesh))
            ret_values.append(ret_value)

    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            (rec[classname], prec[classname], ap[classname]), (rec_mesh[classname], prec_mesh[classname], ap_mesh[classname]) = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0

            rec_mesh[classname] = 0
            prec_mesh[classname] = 0
            ap_mesh[classname] = 0
        print(classname, 'box', ap[classname])
        print(classname, 'mesh', ap_mesh[classname])

    return (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh)

def eval_det_multiprocessing_wo_mesh(pred_all, gt_all, ovthresh=0.25, use_07_metric=True, get_iou_func=get_iou):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    ret_values = p.map(eval_det_cls_wrapper_wo_mesh,
                       [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func) for classname in
                        gt.keys() if classname in pred])
    p.close()
    p.join()
    # ret_values = []
    # for classname in  gt.keys():
    #     if classname in pred:
    #         ret_value = eval_det_cls_wrapper_wo_mesh(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func)
    #         ret_values.append(ret_value)
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
        print(classname, ap[classname])

    return rec, prec, ap

def eval_det_multiprocessing_wo_mesh2(pred_all, gt_all, ovthresh=0.25, use_07_metric=True, get_iou_func=get_iou):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    gt_boxids = {}
    for batch_id in pred_all.keys():
        for scan_name, classname, bbox, score in pred_all[batch_id]:
            if classname not in pred: pred[classname] = {}
            if scan_name not in pred[classname]:
                pred[classname][scan_name] = []
            pred[classname][scan_name].append((bbox, score))
    for batch_id in gt_all.keys():
        for scan_name, boxid, classname, bbox in gt_all[batch_id]:
            if classname not in gt: gt[classname] = {}
            if scan_name not in gt[classname]:
                gt[classname][scan_name] = []
            if scan_name not in gt_boxids:
                gt_boxids[scan_name] = []
            if boxid not in gt_boxids[scan_name]:
                gt[classname][scan_name].append(bbox)
                gt_boxids[scan_name].append(boxid)
    # for batch_id in gt_all.keys():
    #     for scan_name, classname, bbox in gt_all[batch_id]:
    #         if classname not in gt: gt[classname] = {}
    #         if scan_name not in gt[classname]:
    #             gt[classname][scan_name] = []
    #         gt[classname][scan_name].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    ret_values = p.map(eval_det_cls_wrapper_wo_mesh,
                       [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func) for classname in
                        gt.keys() if classname in pred])
    p.close()
    p.join()
    # ret_values = []
    # for classname in  gt.keys():
    #     if classname in pred:
    #         ret_value = eval_det_cls_wrapper_wo_mesh(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func)
    #         ret_values.append(ret_value)
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
        print(classname, ap[classname])

    return rec, prec, ap


def eval_det_cls_distance(pred, gt, thresh, use_07_metric, distance_method = sample_and_calc_chamfer):
    # per-class! pred = {scan_id: [bbox, score, mesh]}

    # construct gt
    npos = 0
    all_stats = {}  # {scan_id: {'bbox': bbox list, 'det': matched list}}
    for scan_id in gt.keys():
        mesh = gt[scan_id]
        matched = [False] * len(mesh)
        npos += len(mesh)
        all_stats[scan_id] = {'meshes_gt': mesh, 'matched': matched}

    # if pred has a scan_id not in gt, also add a dummy gt.
    for scan_id in pred.keys():
        if scan_id not in gt:
            all_stats[scan_id] = {'meshes_gt': [], 'matched': []}

    # construct preds
    scan_ids = []
    confidence = []
    meshes_pred = []

    for scan_id in pred.keys():
        for mesh, score in pred[scan_id]:
            scan_ids.append(scan_id)
            meshes_pred.append(mesh)
            confidence.append(score)

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    meshes_pred = [meshes_pred[x] for x in sorted_ind]
    scan_ids = [scan_ids[x] for x in sorted_ind]

    # go down preds and mark TPs and FPs
    nd = len(meshes_pred)  # number of predicted meshes
    tp_mesh_list = np.zeros(nd)  # 0/1 if TP
    fp_mesh_list = np.zeros(nd)  # 0/1 if FP
    tp_mesh_iou = np.zeros(nd)  # float, the best IoU if TP

    # for all detected bboxes
    for d in range(nd):
        # if d % 100 == 0: print(d)
        print('[eval mesh]', d + 1, '/', nd)

        mesh_pred = meshes_pred[d]

        # compare with per-GT
        best_val = np.inf
        stats = all_stats[scan_ids[d]]  # current scan's stats
        meshes_gt = stats['meshes_gt']  # all gt meshes in this scan
        if len(meshes_gt) > 0:
            # compute overlaps
            for j in range(len(meshes_gt)):

                # metric is implemented here.
                val = distance_method(mesh_pred, meshes_gt[j])
                if val < best_val:
                    best_val = val
                    best_idx = j

            print(f'[eval mesh] best mesh CD {best_val}, from {best_idx} in {j}')

        if best_val < thresh and not stats['matched'][best_idx]:
            tp_mesh_list[d] = 1
            tp_mesh_iou[d] = best_val
            stats['matched'][best_idx] = 1
            print(f'[eval mesh] mesh TP ++')
        else:
            fp_mesh_list[d] = 1

    # for mesh
    fp_mesh_cumsum = np.cumsum(fp_mesh_list)
    tp_mesh_cumsum = np.cumsum(tp_mesh_list)
    rec_mesh = tp_mesh_cumsum / np.maximum(npos, np.finfo(np.float64).eps)
    prec_mesh = tp_mesh_cumsum / np.maximum(tp_mesh_cumsum + fp_mesh_cumsum, np.finfo(np.float64).eps)
    ap_mesh = voc_ap(rec_mesh, prec_mesh, use_07_metric)

    tp_mesh = tp_mesh_cumsum[-1]
    fp_mesh = fp_mesh_cumsum[-1]
    fn_mesh = npos - tp_mesh
    RQ_mesh = tp_mesh / (tp_mesh + fp_mesh / 2 + fn_mesh / 2)
    SQ_mesh = np.sum(tp_mesh_iou) / np.maximum(tp_mesh, np.finfo(np.float64).eps)
    PQ_mesh = SQ_mesh * RQ_mesh

    print(f'[eval mesh] tp = {tp_mesh}, fp = {fp_mesh}, fn = {fn_mesh}, npos = {npos}')

    return (rec_mesh[-1], prec_mesh[-1], ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh)

def eval_det_cls_wrapper_distance(arguments):
    # pred: {scan_id: (bbox, score, mesh)}
    # gt: {scan_id: (bbox, mesh)}
    pred, gt, thresh, use_07_metric, distance_method = arguments
    (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh) = eval_det_cls_distance(pred, gt, thresh, use_07_metric,distance_method)
    return (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh)

# RfD use 07
def eval_det_multiprocessing_distance(pred_all, gt_all, thresh=0.25, use_07_metric=True, distance_method_name = 'cd'):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {scan_id: [(label, bbox, score, mesh)]}
            gt_all: map of {scan_id: [(label, bbox, mesh)]}
            thresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {label: rec}
            prec: {label: prec_all}
            ap: {label: scalar}
    """
    pred = {}  # map {label: pred}
    gt = {}  # map {label: gt}

    if distance_method_name == 'cd':
        distance_method = sample_and_calc_chamfer
    elif distance_method_name == 'lfd':
        distance_method = calc_lfd
    else:
        print(f'uknown distance method name {distance_method_name}\n  use chamfer distance as default')
        distance_method = sample_and_calc_chamfer
    # scan_id == scan_id
    for scan_id in pred_all.keys():
        for label, _, score, _, mesh in pred_all[scan_id]: #classname, bbox, score, voxels, mesh
            # default dict behaviour
            if label not in pred: pred[label] = {}
            if scan_id not in pred[label]: pred[label][scan_id] = []
            if label not in gt: gt[label] = {}
            if scan_id not in gt[label]: gt[label][scan_id] = []
            # record
            pred[label][scan_id].append((mesh, score))

    for scan_id in gt_all.keys():
        for label, _, _, mesh in gt_all[scan_id]: #classname, bbox, score, voxels, mesh
            # default dict behaviour
            if label not in gt: gt[label] = {}
            if scan_id not in gt[label]: gt[label][scan_id] = []
            # record
            gt[label][scan_id].append(mesh)

    rec_mesh = {}
    prec_mesh = {}
    ap_mesh = {}

    PQ_mesh = {}
    SQ_mesh = {}
    RQ_mesh = {}

    # parallel for all classes
    # p = Pool(processes=8)
    # ret_values = p.map(eval_det_cls_wrapper_w_mesh, [(pred[label], gt[label], thresh, use_07_metric, get_iou_func, get_iou_mesh) for label in gt.keys() if label in pred])
    # p.close()
    # p.join()

    # fallback to single-thread
    ret_values = []
    for label in gt.keys():
        if label not in pred: continue
        # print(f'[eval mesh] class {CAD_labels[label]}')
        ret_value = eval_det_cls_wrapper_distance((pred[label], gt[label], thresh, use_07_metric, distance_method))
        ret_values.append(ret_value)

    for i, label in enumerate(gt.keys()):
        if label in pred:
            (rec_mesh[label], prec_mesh[label], ap_mesh[label]), (PQ_mesh[label], SQ_mesh[label], RQ_mesh[label]) = \
            ret_values[i]
        else:
            (rec_mesh[label], prec_mesh[label], ap_mesh[label]), (PQ_mesh[label], SQ_mesh[label], RQ_mesh[label]) = \
                (0,0,0),(0,0,0)

    return (rec_mesh, prec_mesh, ap_mesh), (PQ_mesh, SQ_mesh, RQ_mesh)