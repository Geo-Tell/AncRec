import pickle
import numpy as np
import os
from glob import glob
import re
import open3d as o3d
import copy
from configs.path_config import ScanNet_OBJ_CLASS_NAMES, shapenetid2class
import trimesh
from configs.path_config import scannet_processed_path
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path as shapenet_path


def reverse_trimesh_faces(input_pth, output_pth):
    tmesh = trimesh.load(input_pth)
    triangles = []
    for face in tmesh.faces:
        if len(face) == 3:
            triangles.append(list(face[::-1]))
    tmesh_new = trimesh.Trimesh(tmesh.vertices, np.array(triangles), process=False)
    tmesh_new.export(output_pth)

def flip_trimesh_by_x(input_pth, output_pth):
    tmesh = trimesh.load(input_pth)
    vertices_original = []
    vertices = []
    for vertex in tmesh.vertices:
        vertices_original.append(list(vertex))
        vertices.append(list([-vertex[0],vertex[1],vertex[2]]))
    vertices_x_center = np.mean(np.array(vertices_original),0)[0]
    # = center - (vertex[0]-center)
    # vertices_ = []
    for v in vertices:
        v_new = v
        v_new[0] = v[0] + 2 * vertices_x_center
        # vertices_.append(v_new)

    tmesh_new = trimesh.Trimesh(np.array(vertices), tmesh.faces, process=False)
    tmesh_new.export(output_pth)


def save_scene_mesh(root_path,scene_name ,output_path, align_to_gt=True):
    ##todo data reformatting
    # scene_file_name (scene_name)
    # scene_name = '_'.join(scene_file_name.split('_')[2:])
    # scene_test_number =  scene_file_name.split('_')[1]
    # pth = os.path.join(root_path, scene_file_name, 'test'+ scene_test_number + '_' + scene_name)
    pth = os.path.join(root_path, scene_name)
    temp_pth = os.path.join(root_path, 'temp' , scene_name)
    if not os.path.exists(temp_pth):
        os.makedirs(temp_pth)

    # pc = np.load(os.path.join(pth, 'pc.npy'))
    vis_bbox = np.load(os.path.join(pth, 'vis_bbox.npy'))
    proposal_map = np.load(os.path.join(pth, 'proposal_ids.npy'))
    obj_probs = np.load(os.path.join(pth,'obj_probs.npy'))

    # utils.pc_util.write_ply(pc[:,:3],os.path.join(output_path, scene_name+'_pc.ply'))

    # transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    if align_to_gt:
        pred_mesh_paths = [f for f in os.listdir(pth) if
                           re.search(r'target_(\d+)_proposal_(\d+)_class_(\d+)_mesh.ply', f)]
    else:
        pred_mesh_paths = [f for f in os.listdir(pth) if
                           re.search(r'proposal_(\d+)_class_(\d+)_mesh.ply', f)]
    # proposal_ids = sorted([int(f[:-4].split('_')[3]) for f in pred_mesh_paths])

    for mesh_file in pred_mesh_paths:
        '''Fit obj points to bbox'''
        if align_to_gt:
            target_id, proposal_id, cls_id = \
                re.findall(r'target_(\d+)_proposal_(\d+)_class_(\d+)_mesh.ply', mesh_file)[0]
        else:
            proposal_id, cls_id = \
                re.findall(r'proposal_(\d+)_class_(\d+)_mesh.ply', mesh_file)[0]
        bbox_param = vis_bbox[list(proposal_map).index(int(proposal_id))]
        # bbox_param = bbox_params[list(proposal_map[:, 0]).index(int(proposal_id))]
        center = bbox_param[:3]
        # center[2] = -center[2]
        orientation = bbox_param[6]+ np.pi
        sizes = bbox_param[3:6]

        # mesh_file_pth = os.path.join(pth, mesh_file)
        # # mesh = trimesh.load(mesh_file_pth)
        # reverse_trimesh_faces(mesh_file_pth,  os.path.join(temp_pth, mesh_file))
        # mesh = o3d.io.read_triangle_mesh(os.path.join(temp_pth, mesh_file))
        mesh = o3d.io.read_triangle_mesh(os.path.join(pth, mesh_file))
        ## if directory name contains "test_xx",do the following
        # scene_name = '.'.join(scene_name.split('_')[1:])
        obj_prob = obj_probs[list(proposal_map).index(int(proposal_id))]
        # obj_prob = obj_probs[list(proposal_map[:,0]).index(int(proposal_id))]
        mesh_file_name = os.path.join(output_path, scene_name+'_'+str(proposal_id)+'_'+ ScanNet_OBJ_CLASS_NAMES[int(cls_id)]+'_'+ '%.4f'%obj_prob +'.ply')        # mesh_file_name = os.path.join(new_mesh_file_dir, '.'.join([mesh_file.split('/')[-1].split('.')[0],'.obj']))
        # transform_m = np.array([[1, 0, 0, 0], [0, 0, 1,1], [0, -1, 0,0],[0, 0, 0,1]])
        # transform_m = np.array([[0, 0, 1, 0], [1, 0, 0,0], [0, -1, 0,0],[0, 0, 0,1]])
        transform_m = np.array([[0, 0, -1, 0], [-1, 0, 0,0], [0, 1, 0, 0],[0, 0, 0, 1]])
        mesh_t = copy.deepcopy(mesh)
        mesh_t.transform(transform_m)

        obj_points = np.asarray(mesh_t.vertices)
        mesh_center = (obj_points.max(0) + obj_points.min(0))/2.
        mesh_t.translate(-mesh_center)
        mesh_size = np.zeros(3)
        for i in range(len(sizes)):
            mesh_size[i] = sizes[i] / (obj_points.max(0)[i] - obj_points.min(0)[i])

        scale_T = np.eye(4)
        scale_T[:3,:3] = np.diag(mesh_size)
        mesh_t.transform(scale_T)

        rotation = [-np.sin(orientation/2).tolist(), 0.0,0.0,  np.cos(orientation/2).tolist()]
        R = mesh_t.get_rotation_matrix_from_quaternion(np.array(rotation))
        mesh_t.rotate(R, center=mesh_center)
        mesh_t.translate(center)
        o3d.io.write_triangle_mesh(mesh_file_name, mesh_t)

def save_gt_scene_mesh(scene_names, output_path):
    # scene_names = os.listdir(scannet_processed_path).sort()
    for scene_name in scene_names:
        print(scene_name)
        if not os.path.exists(os.path.join(scannet_processed_path, scene_name, 'bbox.pkl')):
            print('%s doesn\'t exist' %scene_name)
            continue
        with open(os.path.join(scannet_processed_path, scene_name, 'bbox.pkl'), 'rb') as file:
            box_info = pickle.load(file)
        num_bboxes = len(box_info)
        for obj_id in range(num_bboxes):
            box3d = box_info[obj_id]['box3D']
            cls_id_ = int(box_info[obj_id]['cls_id'])
            cls_id = shapenetid2class[cls_id_]
            center  = box3d[:3]
            sizes = box3d[3:6]
            orientation = box3d[-1] + np.pi
            shapenet_id = box_info[obj_id]['shapenet_id']
            shapenet_category_id = box_info[obj_id]['shapenet_catid']
            mesh = o3d.io.read_triangle_mesh(os.path.join(shapenet_path, shapenet_category_id, shapenet_id+'.off'))
            mesh_file_name = os.path.join(output_path,
                                          scene_name + '_' + str(obj_id) + '_' + ScanNet_OBJ_CLASS_NAMES[cls_id]+ '.ply')
            transform_m = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            mesh_t = copy.deepcopy(mesh)
            mesh_t.transform(transform_m)

            obj_points = np.asarray(mesh_t.vertices)
            mesh_center = (obj_points.max(0) + obj_points.min(0)) / 2.
            mesh_t.translate(-mesh_center)
            mesh_size = np.zeros(3)
            for i in range(len(sizes)):
                mesh_size[i] = sizes[i] / (obj_points.max(0)[i] - obj_points.min(0)[i])

            scale_T = np.eye(4)
            scale_T[:3, :3] = np.diag(mesh_size)
            mesh_t.transform(scale_T)

            rotation = [-np.sin(orientation / 2).tolist(), 0.0, 0.0, np.cos(orientation / 2).tolist()]
            R = mesh_t.get_rotation_matrix_from_quaternion(np.array(rotation))
            mesh_t.rotate(R, center=mesh_center)
            mesh_t.translate(center)
            o3d.io.write_triangle_mesh(mesh_file_name, mesh_t)

            #/home/dmy/data/datasets/scannet/processed_data/scene0706_00/bbox.pkl

def save_pred_scene_mesh(pth, mesh_dir, output_pth):
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)

    with open(pth, 'rb') as pf:
        data = pickle.load(pf)
    shape_cls_id = 0
    total_iter = 0
    for iter, shapenet_id in enumerate(data.keys()):
        shape_cls_id += 1
        shape_count = len(data[shapenet_id]['input_features'])
        box_parameters_list = data[shapenet_id]['bboxes']
        scan_name_list = data[shapenet_id]['scan_names']
        obj_id_list = data[shapenet_id]['box_ids']
        proposal_id_list = data[shapenet_id]['proposal_ids']

        for i in np.arange(shape_count):
            scan_name = scan_name_list[i]
            obj_id = obj_id_list[i]
            mesh_pth = 'iter_%s_%s_obj_%s_(label_%s)_pred.ply'% (total_iter, scan_name, obj_id, shape_cls_id)
            if len(glob(os.path.join(mesh_dir, mesh_pth)))==0:
                continue

            total_iter += 1
            box_parameters = box_parameters_list[i]
            proposal_id = proposal_id_list[i]

            center = box_parameters[:3]
            sizes = box_parameters[3:6]
            orientation = box_parameters[6] + np.pi
            obj_prob = box_parameters[-1]

            # pred_cls = proposal_id[2] #gt_cls
            pred_cls_probs = box_parameters[7:15]
            pred_cls = np.argmax(pred_cls_probs, -1)  # B,num_proposal
            pred_cls_prob = pred_cls_probs[pred_cls]
            # iou_with_gt = data[shapenet_id]['iou_with_gt'][i]
            # sampling_number_per_anchor = data[shapenet_id]['sampling_number_per_anchor'][i]
            # bad_anchor_number = (sampling_number_per_anchor < 1).sum()

            pred_scene_mesh_pth  =  glob(os.path.join(mesh_dir, mesh_pth))[0]
            print(pred_scene_mesh_pth)
            if not os.path.exists(pred_scene_mesh_pth):
                print('error: path not exists')

            mesh = o3d.io.read_triangle_mesh(pred_scene_mesh_pth)
            mesh_file_name = os.path.join(output_pth,
                                          scan_name + '_' + str(proposal_id[0]) + '_' + ScanNet_OBJ_CLASS_NAMES[
                                              int(pred_cls)] + '_' + '%.4f' % obj_prob + '_' + '%.4f' % pred_cls_prob + '.ply')  # mesh_file_name = os.path.join(new_mesh_file_dir, '.'.join([mesh_file.split('/')[-1].split('.')[0],'.obj']))
            transform_m = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            mesh_t = copy.deepcopy(mesh)
            mesh_t.transform(transform_m)

            obj_points = np.asarray(mesh_t.vertices)
            if len(obj_points)==0:
                print('error: no vertex in mesh')
                continue
            mesh_center = (obj_points.max(0) + obj_points.min(0)) / 2.
            mesh_t.translate(-mesh_center)
            mesh_size = np.zeros(3)
            for i in range(len(sizes)):
                mesh_size[i] = sizes[i] / (obj_points.max(0)[i] - obj_points.min(0)[i])

            scale_T = np.eye(4)
            scale_T[:3, :3] = np.diag(mesh_size)
            mesh_t.transform(scale_T)

            rotation = [-np.sin(orientation / 2).tolist(), 0.0, 0.0, np.cos(orientation / 2).tolist()]
            R = mesh_t.get_rotation_matrix_from_quaternion(np.array(rotation))
            mesh_t.rotate(R, center=mesh_center)
            mesh_t.translate(center)
            o3d.io.write_triangle_mesh(mesh_file_name, mesh_t)

def transform_mesh_to_scene(mesh, scan_name, box_id, box_parameters, output_dir, mesh_file_name = None):
    center = box_parameters[:3]
    sizes = box_parameters[3:6]
    orientation = box_parameters[6] + np.pi
    obj_prob = box_parameters[-1]
    pred_cls_probs = box_parameters[7:15]
    pred_cls = np.argmax(pred_cls_probs, -1)  # B,num_proposal
    pred_cls_prob = pred_cls_probs[pred_cls]
    if mesh_file_name is None:
        mesh_file_name = os.path.join(output_dir,
                                      scan_name + '_' + str(box_id) + '_' + ScanNet_OBJ_CLASS_NAMES[
                                          int(pred_cls)] + '_' + '%.4f' % obj_prob + '_' + '%.4f' % pred_cls_prob + '.ply')  # mesh_file_name = os.path.join(new_mesh_file_dir, '.'.join([mesh_file.split('/')[-1].split('.')[0],'.obj']))
    transform_m = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    mesh_t = copy.deepcopy(mesh)
    mesh_t.transform(transform_m)
    obj_points = np.asarray(mesh_t.vertices)
    if len(obj_points)==0:
        print('error: no vertex in mesh')
        return
    mesh_center = (obj_points.max(0) + obj_points.min(0)) / 2.
    mesh_t.translate(-mesh_center)
    mesh_size = np.zeros(3)
    for i in range(len(sizes)):
        mesh_size[i] = sizes[i] / (obj_points.max(0)[i] - obj_points.min(0)[i])
    scale_T = np.eye(4)
    scale_T[:3, :3] = np.diag(mesh_size)
    mesh_t.transform(scale_T)
    rotation = [-np.sin(orientation / 2).tolist(), 0.0, 0.0, np.cos(orientation / 2).tolist()]
    R = mesh_t.get_rotation_matrix_from_quaternion(np.array(rotation))
    mesh_t.rotate(R, center=mesh_center)
    mesh_t.translate(center)
    o3d.io.write_triangle_mesh(mesh_file_name, mesh_t)
    return

if __name__ == '__main__':
    # pth = './out/singleObject/new_dataloader/7/test/selected'
    # output_pth = './out/singleObject/new_dataloader/7/test/mesh_in_scene'

    # pth = '/home/dmy/indoor3D/myResearch/BSPInScene/out/singleObject/new_dataloader/7/test/conf_thresh0.56/visualization'
    # output_pth = './out/best_mesh_in_scene'
    # pth = 'out/prepare_data/102_median/test/data.pkl'
    # mesh_dir = 'out/align/new_fusion_no_objectness_3xloss/1/test/1/visualization'
    # mesh_dir = 'out/align_ExponentialLR/1/test/test_best_total/visualization'
    # output_pth = 'out/align_ExponentialLR/1/test/test_best_total/mesh_in_scene2'
    # mesh_dir = 'out/align/new_fusion_no_objectness/1/test/1/visualization'
    # mesh_dir = 'out/align/new_fusion_no_objectness_3xloss/1/test/1/visualization'
    # output_pth = 'out/align/new_fusion_no_objectness/1/test/1/mesh_in_scene'

    # pth = 'out/prepare_data/18_median_new/test/data.pkl'
    # mesh_dir = 'out/align_local_mix/PCTransformer/1/test/1/visualization'
    # output_pth = 'out/align_local_mix/PCTransformer/1/test/1/mesh_in_scene'
    # save_pred_scene_mesh(pth, mesh_dir, output_pth)
    # mode = 'train'
    # with open('/home/dmy/data/datasets/scannet/splits/scannet/scannetv2_%s.txt'%mode) as f:
    #     scene_names = f.readlines()
    # scene_names = [scene_name.strip('\n') for scene_name in scene_names]
    scene_names = ['scene0073_00','scene0073_01','scene0073_02','scene0073_03']
    output_pth = 'gt_meshes'
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    save_gt_scene_mesh(scene_names,output_pth)
    #sudo mount 192.168.210.155:/data/dmy/scannet/gt_meshes /home/dmy/data/datasets/scannet/gt_mesh_in_scene

    # with open('/home/dmy/indoor3D/myResearch/BSPInScene/names.pkl', "rb") as pf:
    #     get_test_number = pickle.load(pf)

    # scene_names = os.listdir(pth) #no testxxx_
    # # scene_name = 'scene0599_02'
    # # scene_names = ['scene0690_00']
    # for scene_name in scene_names:
    #     # input_scene_name = '_'.join([get_test_number[scene_name], scene_name])
    #     input_scene_name = scene_name
    #     save_scene_mesh(pth, scene_name, output_pth)
