from plyfile import PlyData, PlyElement
import numpy as np
import torch
import cv2

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def vis_projected_points(points, img,  outfile, pc=None, pointwidth = 1, color = 'r'):
    if color == 'r':
        c = (0, 0, 255)
    else:
        c = (0, 255, 0)
    for pt in points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), pointwidth, c, thickness=-1)

    mins = points.min(0)
    maxs = points.max(0)
    if len(img.shape)==3:
        w, h, _ = img.shape
    else:
        w, h = img.shape
    margin = 10
    minx = max(mins[0] - margin, 0)
    miny = max(mins[1] - margin, 1)
    maxx = min(maxs[0] + margin, h - 1)
    maxy = min(maxs[1] + margin, w - 1)

    cv2.rectangle(img, (int(minx), int(miny)), (int(maxx), int(maxy)), (0, 0, 255), 8)  # left top; right bottom
    cv2.imwrite(outfile+'.jpg', img)
    if pc is not None:
        write_ply(pc, outfile + '_selected.ply')


def crop_image(image, unique_pts2d, pts2d, margin=0, rescale_ratio=1):
    h, w, _ = image.shape
    mins = np.min(unique_pts2d, 0).astype(np.float32)
    maxs = np.max(unique_pts2d, 0).astype(np.float32)
    center = (mins + maxs) / 2
    new_unique_pts2d = unique_pts2d - center
    new_pts2d = pts2d - center
    if abs(rescale_ratio - 1) > 1e-2:
        rescaled_half_range = (maxs - center) * rescale_ratio
        maxs = center + rescaled_half_range
        mins = center - rescaled_half_range
        new_unique_pts2d = new_unique_pts2d * rescale_ratio
        new_pts2d = new_pts2d * rescale_ratio

    minx = max(mins[0] - margin, 0)
    miny = max(mins[1] - margin, 1)
    maxx = min(maxs[0] + margin, w - 1)
    maxy = min(maxs[1] + margin, h - 1)
    new_center_x = (minx + maxx) / 2
    new_center_y = (miny + maxy) / 2
    new_unique_pts2d  = new_unique_pts2d + np.array([new_center_x, new_center_y])
    new_pts2d  = new_pts2d + np.array([new_center_x, new_center_y])

    new_unique_pts2d = new_unique_pts2d.astype(np.float32)
    new_pts2d = new_pts2d.astype(np.float32)

    return image[int(miny): int(maxy + 1), int(minx): int(maxx + 1)], \
           new_unique_pts2d, new_pts2d, np.int32(minx), np.int32(miny), np.int32(maxx+1), np.int32(maxy+1)

def get_random_range(image, unique_pts2d, margin, shift_range=1., scale_ratio=1.):
    h, w, _ = image.shape
    mins = np.min(unique_pts2d, 0).astype(np.float32)- np.array([margin,margin])
    maxs = np.max(unique_pts2d, 0).astype(np.float32)+ np.array([margin,margin])
    center = (mins + maxs) / 2. + shift_range
    window_range = (maxs - mins)/2.
    window_range_rescaled = window_range * scale_ratio
    new_maxs = center + window_range_rescaled
    new_mins = center - window_range_rescaled
    minx = max(new_mins[0], 0)
    miny = max(new_mins[1], 0)
    maxx = min(new_maxs[0], w - 1)
    maxy = min(new_maxs[1], h - 1)
    new_window_range = int(minx), int(miny), int(maxx), int(maxy)
    return new_window_range #, window_range

def adjust_image_range(image, unique_pts2d_range, margin, shift_range=0., scale_ratio=1.):
    h, w, _ = image.shape
    mins = unique_pts2d_range[:2]- np.array([margin,margin])
    maxs = unique_pts2d_range[2:]+ np.array([margin,margin])
    center = (mins + maxs) / 2. + shift_range
    window_range = (maxs - mins)/2.
    window_range_rescaled = window_range * scale_ratio
    new_maxs = center + window_range_rescaled
    new_mins = center - window_range_rescaled
    minx = max(new_mins[0], 0)
    miny = max(new_mins[1], 0)
    maxx = min(new_maxs[0], w - 1)
    maxy = min(new_maxs[1], h - 1)
    new_window_range = int(minx), int(miny), int(maxx), int(maxy)
    return new_window_range #, window_range


def rescale_pts2d(pts2d, pts2d_mask, window_range, target_size):
    pts2d = pts2d.astype(np.float32)
    minx, miny, maxx, maxy = window_range
    w = (maxx - minx)
    h = (maxy - miny)
    # left_top = np.array([minx,miny])
    # new_pts2d = pts2d - left_top
    new_pts2d = np.zeros((len(pts2d), 2),np.float32)
    new_pts2d[:,0] = pts2d[:,0] * target_size[0]/w
    new_pts2d[:,1] =  pts2d[:,1] * target_size[1]/h
    new_left_top = np.array([minx* target_size[0]/w, miny* target_size[1]/h])
    new_pts2d = new_pts2d - new_left_top
    new_pts2d_valid_mask = (new_pts2d[:,0]<target_size[0])*(new_pts2d[:,1]<target_size[1])* (new_pts2d[:,0] >0) * (new_pts2d[:,1] >0)
    new_pts2d_valid_mask =  new_pts2d_valid_mask *  pts2d_mask #都得是valid的
    return new_pts2d, new_pts2d_valid_mask


def vis_intersection(img, gt_pts2d, pts2d, outfile):
    mins = gt_pts2d.min(0)
    maxs = gt_pts2d.max(0)
    w, h, _ = img.shape
    minx = max(mins[0], 0)
    miny = max(mins[1], 1)
    maxx = min(maxs[0], h - 1)
    maxy = min(maxs[1], w - 1)
    cv2.rectangle(img, (int(minx), int(miny)), (int(maxx), int(maxy)), (0, 255, 0), 3)  # left top; right bottom
    for pt in gt_pts2d:
        cv2.circle(img, (int(pt[0]), int(pt[1])),radius=1,color = (0, 255, 0))

    mins = pts2d.min(0)
    maxs = pts2d.max(0)
    minx = max(mins[0], 0)
    miny = max(mins[1], 1)
    maxx = min(maxs[0], h - 1)
    maxy = min(maxs[1], w - 1)
    cv2.rectangle(img, (int(minx), int(miny)), (int(maxx), int(maxy)), (0, 0,255), 3)  # left top; right bottom
    for pt in pts2d:
        cv2.circle(img, (int(pt[0]), int(pt[1])),radius=3,color = (0, 0,255))

    cv2.imwrite(outfile + '.jpg',img)

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

def shapenet2scannet(shapenet_xyz, box3d):
    # shapenet_xyz: N,3
    # box3d:   6
    transform_m = np.array([[0, 0, -1.0], [-1.0, 0, 0], [0, 1.0, 0]])
    shapenet_xyz = np.dot(transform_m, shapenet_xyz.transpose()).transpose()

    box_xyz = box3d[:3]
    box_size = box3d[3:6]
    box_orientation = box3d[6]  # - np.pi

    # box_size 是在ScanNet的坐标系下定义的，而shapenet_xyz_gt是在shapenet坐标系下，因此要先把shapenet_xyz_gt变换到ScanNet坐标系再计算大小
    rescale = np.zeros(3)
    for i in range(len(rescale)):
        rescale[i] = box_size[i] /(shapenet_xyz.max(0)[i] - shapenet_xyz.min(0)[i])

    mesh_center = (shapenet_xyz.max(0) + shapenet_xyz.min(0)) / 2.
    shapenet_xyz_ = shapenet_xyz - mesh_center
    shapenet_xyz_rescaled = shapenet_xyz_ * rescale

    # align objects to the canonical system.
    rot_matrix = np.zeros([3, 3])
    rot_matrix[0, 0] = np.cos(box_orientation)
    rot_matrix[0, 1] = -np.sin(box_orientation)
    rot_matrix[1, 1] = np.cos(box_orientation)
    rot_matrix[1, 0] = np.sin(box_orientation)
    rot_matrix[2, 2] = 1.
    # obj_xyz_ = obj_xyz_.dot(rot_matrix)

    shapenet_xyz_rotated = np.dot(rot_matrix, shapenet_xyz_rescaled.transpose())
    scannet_xyz = shapenet_xyz_rotated.transpose() + box_xyz

    return scannet_xyz  # N,3



