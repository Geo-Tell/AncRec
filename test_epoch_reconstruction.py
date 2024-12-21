import os
import cv2
import open3d.io
from models.anchorrec.SingleObjectDatasetWithRGB import SingleObject_recon_new, collate_fn,my_worker_init_fn, denormalize_image
from torch.utils.data import DataLoader
import trimesh
from utils.pc_util import write_ply
# import numpy as np
# from configs.path_config import PathConfig
# from net_utils.libs import softmax
# from models.anchorrec.SingleObjectDatasetWithRGB import  SingleObject, collate_fn,my_worker_init_fn, denormalize_image2
from save_scene_mesh import transform_mesh_to_scene
from tqdm import tqdm
def test(cfg,  tester):
    tester.net.train(False)
    vis_path = cfg.config['log']['vis_path']
    scene_mesh_pth = '/'.join(vis_path.split('/')[:-1]) + '/mesh_in_scene'
    os.makedirs(scene_mesh_pth)
    vis_gt = cfg.config['log'].get('vis_gt', False)
    vis_img = cfg.config['log'].get('vis_img', False)
    dataset = SingleObject_recon_new(cfg,'vis') # 'val') #'test') #
    test_loader = DataLoader(dataset = dataset,
                                  num_workers=cfg.config['device']['num_workers'],
                                  batch_size= 1, # cfg.config['test']['batch_size'],
                                  collate_fn=collate_fn,
                                  worker_init_fn=my_worker_init_fn)
    print(len(test_loader))
    for iter, data in tqdm(enumerate(test_loader)):
        data = tester.to_device(data)
        bid = 0
        scan_name = data['scan_name'][bid]
        obj_id = data['box_id'][bid].item()
        shape_label = data['shape_label'][bid].item()
        # input points
        # input_points = data['input_points'][bid].cpu().numpy()
        # output_pts_pth = os.path.join(vis_path, 'iter_%s_%s_obj_%s(label_%s)_pts.ply' % (
        #     iter, scan_name, obj_id, shape_label))
        # write_ply(input_points, output_pts_pth)
        # # input points
        # if "anchors" in data:
        #     input_points = data['anchors'][bid].cpu().numpy()
        #     output_pts_pth = os.path.join(vis_path, 'iter_%s_%s_obj_%s(label_%s)_anchors.ply' % (
        #         iter, scan_name, obj_id, shape_label))
        #     write_ply(input_points, output_pts_pth)
        # gt_mesh_pth = os.path.join(vis_path, '%s_gt.ply' % (shape_label))
        # if not os.path.exists(gt_mesh_pth):
        #     gt_mesh = trimesh.load(data['mesh_pth'][bid], process=False)
        #     gt_mesh.export(gt_mesh_pth)

        est_data = tester.net.module.generate(data)#, vis_gt = True)
        # loss = tester.net.module.loss(est_data, data)
        # output_loss = {}
        # for key, value in loss.items():
        #     if torch.is_tensor(value):
        #         output_loss[key] = loss[key].item()
        #     elif type(value)==float:
        #         output_loss[key] = loss[key]
        #     elif type(value)==int:
        #         output_loss[key] = loss[key]
        # cfg.log_string('%d/%d. Current loss: %s.' % (iter + 1, len(test_loader), str(output_loss)))
        bbox_params = data['bbox_params'].cpu().numpy() if 'bbox_params' in data else None
        box_ids = data['box_id'].cpu().numpy()
        for bid, mesh in enumerate(est_data['meshes']):
            #pred_mesh
            output_mesh_pth = os.path.join(vis_path, 'iter_%s_%s_obj_%s(label_%s)_pred_.ply' % (iter,scan_name, obj_id, shape_label))
            mesh.export(output_mesh_pth)#iter_%s_
            # transform
            # if bbox_params  is not None:
            #     tri_mesh = open3d.io.read_triangle_mesh(output_mesh_pth)
            #     transform_mesh_to_scene(tri_mesh, scan_name, box_ids[bid], bbox_params[bid], scene_mesh_pth)
                #mesh, scan_name, box_id, box_parameters, output_dir

            # input pts
            # write_ply(data['object_points'][bid].detach().cpu().numpy(), os.path.join(vis_path, 'iter_%s_%s_obj_%s_(label_%s)_objpts.ply' % (iter,scan_name, obj_id, shape_label)))
            if vis_img and 'img' in data:
                cv2.imwrite( os.path.join(vis_path, 'iter_%s_%s_obj_%s_(label_%s)_input_img.jpg' % (iter,scan_name, obj_id, shape_label)),\
                            data['img'][bid].cpu().numpy()) #iter%s_
                            # denormalize_image(data['img'][bid])) #iter%s_

            # if vis_gt and 'shape_label' in data:
            #     shape_label = data['shape_label'][bid].item()
            #     gt_z_mesh_pth = os.path.join(vis_path, '%s_gt_z.ply' % (shape_label))
            #     if not os.path.exists(gt_z_mesh_pth) and len(est_data['gt_meshes'][bid])>0:
            #         est_data['gt_meshes'][bid].export(gt_z_mesh_pth)
            #     gt_mesh_pth = os.path.join(vis_path, '%s_gt.ply' % (shape_label))
            #     if not os.path.exists(gt_mesh_pth):
            #         gt_mesh = trimesh.load(data['mesh_pth'][bid], process=False)
            #         gt_mesh.export(gt_mesh_pth)

        # for bid in range(data['original_object_points'].shape[0]):
        #     scan_name = data['scan_name'][bid]
        #     obj_id = data['obj_id'][bid].item()
        #     write_ply(data['original_object_points'][bid].detach().cpu().numpy(), os.path.join(vis_path,
        #             '%s_obj_%s_input_original_points.ply'% (scan_name, obj_id)))
        #     write_ply(data['object_points'][bid].detach().cpu().numpy(), os.path.join(vis_path,
        #             '%s_obj_%s_input_points.ply' % (scan_name, obj_id)))

            # input point cloud (save only once for each scene)
            # with open(os.path.join(vis_path,'iter_%s_bid_%s_%s.txt' % (iter, bid,data['scan_name'][bid]))):
            #     pass
            # scene_pc = np.load(os.path.join(PathConfig.processed_data_path, data['scan_name'][bid], 'full_scan.npz'))['mesh_vertices'][:,:3]
            # output_scene_pc_path = os.path.join(vis_path, '%s.ply' % (data['scan_name'][bid]))
            # if not os.path.exists(output_scene_pc_path):
            #     write_ply(scene_pc, output_scene_pc_path)

    # np.save(os.path.join(vis_path,'objectness_scores.npy'),np.array(objectness_scores_list))
    # np.save(os.path.join(vis_path,'iou_with_gt.npy'),np.array(iou_with_gt_list))



def test_img(cfg,  tester):
    tester.net.train(False)
    vis_path = cfg.config['log']['vis_path']
    scene_mesh_pth = '/'.join(vis_path.split('/')[:-1]) + '/imgs'
    os.makedirs(scene_mesh_pth)
    bid = 0
    dataset = SingleObject_recon_new(cfg, 'test')
    test_loader = DataLoader(dataset = dataset,
                                  num_workers=cfg.config['device']['num_workers'],
                                  batch_size=1,
                                  collate_fn=collate_fn,
                                  worker_init_fn=my_worker_init_fn)
    for iter, data in enumerate(test_loader):
        scan_name = data['scan_name'][bid]
        obj_id = data['box_id'][bid].item()
        shape_label = data['shape_label'][bid].item()
        cv2.imwrite( os.path.join(vis_path, 'iter_%s_%s_obj_%s_(label_%s)_input_img.jpg' % (iter,scan_name, obj_id, shape_label)),\
                    denormalize_image(data['img'][bid])) #iter%s_
