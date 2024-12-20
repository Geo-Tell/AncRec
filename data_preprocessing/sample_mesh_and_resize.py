import pickle
import argparse
import os
import trimesh
import numpy as np
from configs.path_config import scannet_processed_path
from utils.shapenet import ShapeNetv2_Watertight_path

def parse_args():
    '''Parameters'''
    parser = argparse.ArgumentParser('Prepare ShapeNetv2 Data.')
    parser.add_argument('--float16', action='store_true',
                        help='Whether to use half precision.')
    parser.add_argument('--points_size', type=int, default=1024,
                        help='Size of points.')
    return parser.parse_args()

def main(args):
    dir_list = os.listdir(scannet_processed_path)
    dir_list.sort()
    for scene_dirname in dir_list:
        save_path = os.path.join(scannet_processed_path, scene_dirname)
        filename = os.path.join(save_path, 'object_shape_points.npz')
        if os.path.exists(filename):
            print('shape points exists')
            continue
        print(scene_dirname)
        if not os.path.exists(os.path.join(save_path, 'bbox.pkl')):
            print('%s bbox not exists'%scene_dirname)
            continue
        with open(os.path.join(save_path, 'bbox.pkl'), 'rb') as file:
            bboxes = pickle.load(file)

        objects_points = []

        for box in bboxes:
            '''Normalize angles to [-np.pi, np.pi]'''
            box['box3D'][6] = np.mod(box['box3D'][6] + np.pi, 2 * np.pi) - np.pi
            shapenet_model = os.path.join(ShapeNetv2_Watertight_path, box['shapenet_catid'], box['shapenet_id'] + '.off')
            assert os.path.exists(shapenet_model)
            bbox = box['box3D']
            mesh = trimesh.load(shapenet_model)
            transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
            obj_points, _ = mesh.sample(args.points_size, return_index=True)
            obj_points = obj_points - (obj_points.max(0) + obj_points.min(0)) / 2.
            obj_points = obj_points.dot(transform_m.T)
            obj_points = obj_points.dot(np.diag(1 / (obj_points.max(0) - obj_points.min(0)))).dot(np.diag(bbox[3:6]))
            orientation = bbox[6]
            axis_rectified = np.array(
                [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0],
                 [0, 0, 1]])
            obj_points = obj_points.dot(axis_rectified) + bbox[0:3]

            # Compress
            if args.float16:
                dtype = np.float16
            else:
                dtype = np.float32

            points = obj_points.astype(dtype)
            objects_points.append(points)

        np.savez(filename, points= objects_points)


if __name__ == '__main__':
    args = parse_args()
    main(args)