# Modified from https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py # noqa
import os
import struct
from functools import partial
import imageio
import numpy as np
import torch
import pickle
import cv2
from argparse import ArgumentParser
from configs.path_config import scannet_processed_path, raw_scans_path
DEPTH_THRESH = 20
COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}

COMPRESSION_TYPE_DEPTH = {
    -1: 'unknown',
    0: 'raw_ushort',
    1: 'zlib_ushort',
    2: 'occi_ushort'
}
def load_axis_align_matrix(meta_file):
    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [
                float(x)
                for x in line.rstrip().strip('axisAlignment = ').split(' ')
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    return axis_align_matrix

def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (torch.Tensor | np.ndarray): Points in shape (N, 3)
        proj_mat (torch.Tensor | np.ndarray):
            Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        (torch.Tensor | np.ndarray): Points in image coordinates,
            with shape [N, 2] if `with_depth=False`, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res

class RGBDFrame:
    """Class for single ScanNet RGB-D image processing."""

    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack('f' * 16, file_handle.read(16 * 4)),
            dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(
            struct.unpack('c' * self.color_size_bytes,
                          file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(
            struct.unpack('c' * self.depth_size_bytes,
                          file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        assert compression_type == 'zlib_ushort'
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        assert compression_type == 'jpeg'
        return imageio.imread(self.color_data)

#Find the correspondence between image pixels and points in the point cloud
#Store point indices in ptids_images (list of images of indices)
class SensorData:
    def __init__(self, filename, meta_filename, point_cloud_filename, limit): #, output_path):
        self.version = 4
        self.load(filename, meta_filename, point_cloud_filename, limit) #, output_path)

    def load(self, filename, meta_filename, point_cloud_filename, limit, vis=False, output_path=None):
        point_clouds = np.load(point_cloud_filename)['mesh_vertices']
        axis_align_matrix = load_axis_align_matrix(meta_filename)
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = b''.join(
                struct.unpack('c' * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack(
                'i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack(
                'i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]
            self.frames = []


            if limit > 0 and limit < num_frames:
                index = np.random.choice(
                    np.arange(num_frames), limit, replace=False).tolist()
            else:
                index = list(range(num_frames))
            self.index = index
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                if i not in index:
                    continue
                self.frames.append(frame)

            self.ptids_images = []
            self.img_depths_list = []

            for idx, frame in enumerate(self.frames):
                depth_data = frame.decompress_depth(self.depth_compression_type)
                depth = np.fromstring(
                    depth_data, dtype=np.uint16).reshape(self.depth_height,
                                                         self.depth_width)
                depth = cv2.resize(depth, (self.color_width, self.color_height), interpolation=cv2.INTER_LINEAR)
                #Find visible points by comparing the projection depth and the provided depth map
                extrinsic = frame.camera_to_world
                proj_mat = self.intrinsic_color @ np.linalg.inv(axis_align_matrix @ extrinsic)  # depth2image (depth pts to image)
                pts_2d = points_cam2img(torch.tensor(point_clouds[:, :3]), torch.tensor(proj_mat).to(torch.float32),
                                        with_depth=True).numpy()
                pts_depth = pts_2d[:,-1]  #Projection depth
                # find points in fov
                fov_labels = ((pts_2d[:, 0] < self.color_width)
                            & (pts_2d[:, 0] >= 0)
                            & (pts_2d[:, 1] < self.color_height)
                            & (pts_2d[:, 1] >= 0)
                            & (pts_2d[:, 2] > 0))

                in_image_pt2d_indices = np.where(fov_labels)[0]
                pts_2d = pts_2d[:,:2][fov_labels].astype(np.int32)
                img_inds = - np.ones((self.color_height, self.color_width)).astype(np.int32)
                img_depths = 10 * np.ones((self.color_height, self.color_width)).astype(np.float32) # max depth

                filtered_pts2d = []
                left_pts2d = []
                for ptid, pt_2d in enumerate(pts_2d):
                    pt_depth = pts_depth[fov_labels][ptid]
                    if abs(1000 * pt_depth - depth[pt_2d[1], pt_2d[0]]) <  DEPTH_THRESH:
                        img_depths[pt_2d[1], pt_2d[0]] = pt_depth #
                        img_inds[pt_2d[1], pt_2d[0]] = in_image_pt2d_indices[ptid]  #y,x (image coordinate)
                        filtered_pts2d.append(pt_2d)
                    else:
                        left_pts2d.append(pt_2d)
                '''vis for debugging'''
                if vis:
                    if output_path is None:
                        output_path = 'debug/vis_ptids_images'
                    os.makedirs(output_path,exist_ok=True)
                    rgb = frame.decompress_color(self.color_compression_type)
                    img_out = rgb.copy()
                    for pt in filtered_pts2d:
                        cv2.circle(img_out, (int(pt[0]), int(pt[1])), 2, (0, 0, 255),-1)
                    cv2.imwrite(os.path.join(output_path, '%s_filtered_circles.jpg' % idx), img_out)
                    for pt in left_pts2d:
                        cv2.circle(img_out, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(output_path, '%s_leftout_circles.jpg' % idx), img_out)

                    heatmapshow = None
                    heatmapshow = cv2.normalize(depth, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                dtype=cv2.CV_8U)
                    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(output_path, '%s_depth.jpg' % idx), heatmapshow)

                self.ptids_images.append(img_inds)      #point indices on images
                self.img_depths_list.append(img_depths) #projection depths


    def export_pc_info(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(os.path.join(output_path, 'ptids_images.pkl'), "wb") as fp:
            pickle.dump(self.ptids_images,fp)
        with open(os.path.join(output_path, 'img_depths_list.pkl'), "wb") as fp:
            pickle.dump(self.img_depths_list,fp)
        # with open( os.path.join(output_path, 'pc_pts2d.pkl'), "wb") as fp:
        #     pickle.dump(self.pc_pt2d,fp)
        # with open( os.path.join(output_path, 'pc_depth.pkl'), "wb") as fp:
        #     pickle.dump(self.pc_depth,fp)
        # with open( os.path.join(output_path, 'pc_imgids.pkl'), "wb") as fp:
        #     pickle.dump(self.pc_info,fp)
        np.save(os.path.join(output_path,'index.npy'), self.index)

    def export_depth_images(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            depth_data = self.frames[f].decompress_depth(
                self.depth_compression_type)
            depth = np.fromstring(
                depth_data, dtype=np.uint16).reshape(self.depth_height,
                                                     self.depth_width)
            imageio.imwrite(
                os.path.join(output_path,
                             self.index_to_str(f) + '.png'), depth)

    def export_color_images(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            color = self.frames[f].decompress_color(
                self.color_compression_type)
            imageio.imwrite(
                os.path.join(output_path,
                             self.index_to_str(f) + '.jpg'), color)

    @staticmethod
    def index_to_str(index):
        return str(index).zfill(5)

    @staticmethod
    def save_mat_to_file(matrix, filename):
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%f')

    def export_poses(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            self.save_mat_to_file(
                self.frames[f].camera_to_world,
                os.path.join(output_path,
                             self.index_to_str(f) + '.txt'))

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.save_mat_to_file(self.intrinsic_color,
                              os.path.join(output_path, 'intrinsic.txt'))


def process_scene_with_point(root_path, point_path, limit, idx, output_path):
    """Process single ScanNet scene.

    Extract RGB images, poses and camera intrinsics.
    """
    # point_filename = os.path.join(point_path, idx + '_vert.npy')
    point_filename = os.path.join(point_path, idx, 'full_scan.npz')
    output_path = os.path.join(output_path, idx)
    if not os.path.exists(point_filename):
        print(idx)
        return
    if os.path.exists(output_path):
        return
    filename =os.path.join(root_path, idx, f'{idx}.sens')
    meta_filename = os.path.join(root_path, idx, f'{idx}.txt')

    data = SensorData(filename, meta_filename, point_filename,limit)

    data.export_pc_info(output_path)
    data.export_color_images(output_path)
    data.export_depth_images(output_path)
    data.export_intrinsics(output_path)
    data.export_poses(output_path)


    # 取将所有点云都覆盖的最小图片集合
    # 找出现次数最多的img idx, 该图片包含的点云去掉，再在剩下的里面找出现次数最多的，直到所有点云被去掉
    # 方法二：找出现次数最多的前50个看看效果，（物体点通常出现次数多

def process_directory(root_path, point_path, limit, nproc, output_path):
    print(f'processing {root_path}')
    mmcv.track_parallel_progress(
        func=partial(process_scene_with_point, root_path, point_path,  limit, output_path=output_path),
        tasks=os.listdir(root_path),
        nproc=nproc)


def image2instance(posed_images_path, output_pth):
    scan_names = os.listdir(posed_images_path)
    for scan_name in tqdm(scan_names):
        output_pth_scene = os.path.join(output_pth, scan_name)
        # if os.path.exists(output_pth_scene):
        #     continue
        with open(os.path.join(posed_images_path , scan_name, 'ptids_images.pkl'), 'rb') as f:
            ptids_images = pickle.load(f)
        scan_data = np.load(os.path.join(scannet_processed_path, scan_name, "full_scan.npz"))
        point_instance_labels = scan_data['instance_labels']
        instance_point_ratios = dict()
        output_instance2imageid = dict()
        output_instance_ratio = dict()
        unique_labels_all = np.unique(point_instance_labels)
        in_scene_instance_point_numbers = {unique_label: np.sum(point_instance_labels== unique_label) for unique_label in unique_labels_all}
        for unique_label in unique_labels_all:
            instance_point_ratios[unique_label]  = {}

        for img_id, ptids_image in enumerate(ptids_images):
            ptids_image_flattened = ptids_image.flatten()
            ptids_image_flattened = ptids_image_flattened[ptids_image_flattened>-1]
            image_instance_labels = point_instance_labels[ptids_image_flattened]
            unique_labels_in_image = np.unique(image_instance_labels)
            for unique_label in unique_labels_all:
                if unique_label not in unique_labels_in_image :
                    instance_point_ratios[unique_label][img_id] = 0
                else:
                    instance_point_ratios[unique_label][img_id] = np.sum(image_instance_labels == unique_label) \
                                                                  / in_scene_instance_point_numbers[unique_label]


        for instance_label in unique_labels_all:
            ratio_list = np.array([instance_point_ratios[instance_label][key] for key in instance_point_ratios[instance_label].keys()])
            sorted_indices = np.argsort(-ratio_list)
            sorted_point_ratios = ratio_list[sorted_indices]
            output_instance2imageid[instance_label] = sorted_indices.astype(np.float32)
            output_instance_ratio[instance_label] = sorted_point_ratios.astype(np.float32)

        if not os.path.exists(output_pth_scene):
            os.makedirs(output_pth_scene)
        with open(os.path.join(output_pth_scene , 'output_instance2imageid.pkl'), 'wb') as f:
            pickle.dump(output_instance2imageid, f)
        with open(os.path.join(output_pth_scene , 'output_instance_ratio.pkl'), 'wb') as f:
            pickle.dump(output_instance_ratio, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max-images-per-scene', type=int, default=300)
    parser.add_argument('--nproc', type=int, default=8)
    # parser.add_argument('--output-path', type=str, required=True)
    from configs.path_config import posed_images_path, posed_images_instance_path
    args = parser.parse_args()

    # process train and val scenes
    if os.path.exists(raw_scans_path):
        # For debugging
        # process_scene_with_point(raw_data_path, point_path, 100, 'scene0000_02',  args.output_path)
        process_directory(raw_scans_path, scannet_processed_path, args.max_images_per_scene, args.nproc, posed_images_path)
        image2instance(posed_images_path, posed_images_instance_path)
