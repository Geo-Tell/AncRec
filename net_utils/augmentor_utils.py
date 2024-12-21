import numpy as np
import math
import copy
from net_utils import common_utils
from net_utils.libs import rotz
from utils.scannet.scannet_planes import rotate_quad

def random_flip_along_x(points, gt_votes, gt_surface_points,  gt_boxes, gt_quads, gt_quad_votes, gt_quad_surface_points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 1] = -points[:, 1]
        gt_votes[:, [2, 5, 8]] = -1 * gt_votes[:, [2, 5, 8]]
        gt_quad_votes[:, 1] = -1 * gt_quad_votes[:, 1]

        gt_surface_points[:, 1] = -gt_surface_points[:, 1]
        gt_quad_surface_points[:, 1] = -1 * gt_quad_surface_points[:, 1]


        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]

        gt_quads[:, 1] = -1 * gt_quads[:, 1]
        gt_quads[:, 4] = -1 * gt_quads[:, 4]

        
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
    
    return points, gt_votes, gt_surface_points,  gt_boxes, gt_quads, gt_quad_votes, gt_quad_surface_points


def random_flip_along_y(points, gt_votes, gt_surface_points,  gt_boxes, gt_quads, gt_quad_votes, gt_quad_surface_points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 0] = -points[:, 0]
        gt_votes[:, [1, 4, 7]] = -1 * gt_votes[:, [1, 4, 7]]
        gt_quad_votes[:, 0] = -1 * gt_quad_votes[:,0]

        gt_surface_points[:, 0] = -gt_surface_points[:, 0]
        gt_quad_surface_points[:, 0] = -1 * gt_quad_surface_points[:, 0]

        gt_boxes[:, 0] = -gt_boxes[:, 0]
        # gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        gt_boxes[:, 6] = np.sign(gt_boxes[:, 6]) * np.pi - gt_boxes[:, 6]

        gt_quads[:, 0] = -1 * gt_quads[:, 0]
        gt_quads[:, 3] = -1 * gt_quads[:, 3]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return points, gt_votes, gt_surface_points,  gt_boxes, gt_quads, gt_quad_votes, gt_quad_surface_points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points

def global_rotation_original(points, gt_votes, gt_surface_points,  gt_boxes, gt_quads, gt_quad_votes, gt_quad_surface_points):
    rot_angle = (np.random.random() * np.pi / 2) - np.pi / 4  # -45 ~ +45 degree
    rot_mat = rotz(rot_angle)

    gt_votes_end = np.zeros_like(gt_votes)
    gt_votes_end[:, 1:4] = np.dot(points[:, 0:3] + gt_votes[:, 1:4], np.transpose(rot_mat))
    gt_votes_end[:, 4:7] = np.dot(points[:, 0:3] + gt_votes[:, 4:7], np.transpose(rot_mat))
    gt_votes_end[:, 7:10] = np.dot(points[:, 0:3] + gt_votes[:, 7:10], np.transpose(rot_mat))

    points[:, 0:3] = np.dot(points[:, 0:3], np.transpose(rot_mat))
    gt_surface_points[:, 0:3] = np.dot(gt_surface_points[:, 0:3], np.transpose(rot_mat))
    gt_boxes[:, 0:3] = np.dot(gt_boxes[:, 0:3], np.transpose(rot_mat))
    gt_boxes[:, 6] += rot_angle
    gt_votes[:, 1:4] = gt_votes_end[:, 1:4] - points[:, 0:3]
    gt_votes[:, 4:7] = gt_votes_end[:, 4:7] - points[:, 0:3]
    gt_votes[:, 7:10] = gt_votes_end[:, 7:10] - points[:, 0:3]

    gt_quad_votes[:, 0:3] = np.dot(gt_quad_votes[:, 0:3], np.transpose(rot_mat))
    gt_quad_surface_points[:, 0:3] = np.dot(gt_quad_surface_points[:, 0:3], np.transpose(rot_mat))
    rectangles = rotate_quad(gt_quads, rot_mat)

    '''Normalize angles to [-pi, pi]'''
    gt_boxes[:, 6] = np.mod(gt_boxes[:, 6] + np.pi, 2 * np.pi) - np.pi
    return points, gt_votes, gt_surface_points,  gt_boxes, gt_quads, gt_quad_votes, gt_quad_surface_points

def global_rotation_mmdet3d(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] -= noise_rotation
    
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points


def global_scaling(points, gt_votes, gt_surface_points,  gt_boxes, gt_quads, gt_quad_votes, gt_quad_surface_points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return points, gt_votes, gt_surface_points,  gt_boxes, gt_quads, gt_quad_votes, gt_quad_surface_points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_votes[:, 1:4] *= noise_scale
    gt_votes[:, 4:7] *= noise_scale
    gt_votes[:, :7:10] *= noise_scale
    gt_quad_votes *= noise_scale
    # gt_quad_votes[:, 4:7] *= noise_scale
    # gt_quad_votes[:, :7:10] *= noise_scale
    gt_surface_points *= noise_scale
    gt_quad_surface_points *= noise_scale

    gt_boxes[:, :6] *= noise_scale
    gt_quads[:, :3] *= noise_scale
    gt_quads[:, -2:] *= noise_scale

    return points, gt_votes, gt_surface_points,  gt_boxes, gt_quads, gt_quad_votes, gt_quad_surface_points


def random_image_flip_horizontal(image, depth_map, gt_boxes, calib):
    """
    Performs random horizontal flip augmentation
    Args:
        image: (H_image, W_image, 3), Image
        depth_map: (H_depth, W_depth), Depth map
        gt_boxes: (N, 7), 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        calib: calibration.Calibration, Calibration object
    Returns:
        aug_image: (H_image, W_image, 3), Augmented image
        aug_depth_map: (H_depth, W_depth), Augmented depth map
        aug_gt_boxes: (N, 7), Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
    """
    # Randomly augment with 50% chance
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

    if enable:
        # Flip images
        aug_image = np.fliplr(image)
        aug_depth_map = np.fliplr(depth_map)
        
        # Flip 3D gt_boxes by flipping the centroids in image space
        aug_gt_boxes = copy.copy(gt_boxes)
        locations = aug_gt_boxes[:, :3]
        img_pts, img_depth = calib.lidar_to_img(locations)
        W = image.shape[1]
        img_pts[:, 0] = W - img_pts[:, 0]
        pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
        pts_lidar = calib.rect_to_lidar(pts_rect)
        aug_gt_boxes[:, :3] = pts_lidar
        aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6]

    else:
        aug_image = image
        aug_depth_map = depth_map
        aug_gt_boxes = gt_boxes

    return aug_image, aug_depth_map, aug_gt_boxes


def random_translation_along_x(points, gt_surface_points,  gt_boxes, gt_quads, gt_quad_surface_points, offset_std):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)

    points[:, 0] += offset

    gt_surface_points[:, 0] += offset
    gt_quad_surface_points[:, 0] += offset

    gt_boxes[:, 0] += offset
    gt_quads[:, 0] += offset

    return points, gt_surface_points,  gt_boxes, gt_quads, gt_quad_surface_points

def random_translation_along_y(points, gt_surface_points,  gt_boxes, gt_quads, gt_quad_surface_points, offset_std):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)

    points[:, 1] += offset

    gt_surface_points[:, 1] += offset
    gt_quad_surface_points[:, 1] += offset

    gt_boxes[:, 1] += offset
    gt_quads[:, 1] += offset

    return points, gt_surface_points,  gt_boxes, gt_quads, gt_quad_surface_points

def random_translation_along_z(points, gt_surface_points,  gt_boxes, gt_quads, gt_quad_surface_points, offset_std):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)

    points[:, 2] += offset

    gt_surface_points[:, 2] += offset
    gt_quad_surface_points[:, 2] += offset

    gt_boxes[:, 2] += offset
    gt_quads[:, 2] += offset

    return points, gt_surface_points,  gt_boxes, gt_quads, gt_quad_surface_points


def random_local_translation_along_x(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 0] += offset
        
        gt_boxes[idx, 0] += offset
    
        # if gt_boxes.shape[1] > 7:
        #     gt_boxes[idx, 7] += offset
    
    return gt_boxes, points


def random_local_translation_along_y(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 1] += offset
        
        gt_boxes[idx, 1] += offset
    
        # if gt_boxes.shape[1] > 8:
        #     gt_boxes[idx, 8] += offset
    
    return gt_boxes, points


def random_local_translation_along_z(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 2] += offset
        
        gt_boxes[idx, 2] += offset
    
    return gt_boxes, points


def global_frustum_dropout_top(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    # threshold = max - length * uniform(0 ~ 0.2)
    threshold = np.max(points[:, 2]) - intensity * (np.max(points[:, 2]) - np.min(points[:, 2]))
    
    points = points[points[:, 2] < threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 2] < threshold]
    return gt_boxes, points


def global_frustum_dropout_bottom(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    
    threshold = np.min(points[:, 2]) + intensity * (np.max(points[:, 2]) - np.min(points[:, 2]))
    points = points[points[:, 2] > threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 2] > threshold]
    
    return gt_boxes, points


def global_frustum_dropout_left(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    
    threshold = np.max(points[:, 1]) - intensity * (np.max(points[:, 1]) - np.min(points[:, 1]))
    points = points[points[:, 1] < threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 1] < threshold]
    
    return gt_boxes, points


def global_frustum_dropout_right(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    
    threshold = np.min(points[:, 1]) + intensity * (np.max(points[:, 1]) - np.min(points[:, 1]))
    points = points[points[:, 1] > threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 1] > threshold]
    
    return gt_boxes, points


def local_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        # augs[f'object_{idx}'] = noise_scale
        points_in_box, mask = get_points_in_box(points, box)
        
        # tranlation to axis center
        points[mask, 0] -= box[0]
        points[mask, 1] -= box[1]
        points[mask, 2] -= box[2]
        
        # apply scaling
        points[mask, :3] *= noise_scale
        
        # tranlation back to original position
        points[mask, 0] += box[0]
        points[mask, 1] += box[1]
        points[mask, 2] += box[2]
        
        gt_boxes[idx, 3:6] *= noise_scale
    return gt_boxes, points


def local_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        # augs[f'object_{idx}'] = noise_rotation
        points_in_box, mask = get_points_in_box(points, box)
        
        centroid_x = box[0]
        centroid_y = box[1]
        centroid_z = box[2]
        
        # tranlation to axis center
        points[mask, 0] -= centroid_x
        points[mask, 1] -= centroid_y
        points[mask, 2] -= centroid_z
        box[0] -= centroid_x
        box[1] -= centroid_y
        box[2] -= centroid_z
        
        # apply rotation
        points[mask, :] = common_utils.rotate_points_along_z(points[np.newaxis, mask, :], np.array([noise_rotation]))[0]
        box[0:3] = common_utils.rotate_points_along_z(box[np.newaxis, np.newaxis, 0:3], np.array([noise_rotation]))[0][0]
        
        # tranlation back to original position
        points[mask, 0] += centroid_x
        points[mask, 1] += centroid_y
        points[mask, 2] += centroid_z
        box[0] += centroid_x
        box[1] += centroid_y
        box[2] += centroid_z
        
        gt_boxes[idx, 6] += noise_rotation
        if gt_boxes.shape[1] > 8:
            gt_boxes[idx, 7:9] = common_utils.rotate_points_along_z(
                np.hstack((gt_boxes[idx, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                np.array([noise_rotation])
            )[0][:, 0:2]
    
    return gt_boxes, points


def local_frustum_dropout_top(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (z + dz / 2) - intensity * dz
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 2] >= threshold))]
    
    return gt_boxes, points


def local_frustum_dropout_bottom(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (z - dz / 2) + intensity * dz
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 2] <= threshold))]
    
    return gt_boxes, points


def local_frustum_dropout_left(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (y + dy / 2) - intensity * dy
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 1] >= threshold))]
    
    return gt_boxes, points


def local_frustum_dropout_right(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (y - dy / 2) + intensity * dy
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 1] <= threshold))]
    
    return gt_boxes, points


def get_points_in_box(points, gt_box):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    cx, cy, cz = gt_box[0], gt_box[1], gt_box[2]
    dx, dy, dz, rz = gt_box[3], gt_box[4], gt_box[5], gt_box[6]
    shift_x, shift_y, shift_z = x - cx, y - cy, z - cz
    
    MARGIN = 1e-1
    cosa, sina = math.cos(-rz), math.sin(-rz)
    local_x = shift_x * cosa + shift_y * (-sina)
    local_y = shift_x * sina + shift_y * cosa
    
    mask = np.logical_and(abs(shift_z) <= dz / 2.0, 
                          np.logical_and(abs(local_x) <= dx / 2.0 + MARGIN, 
                                         abs(local_y) <= dy / 2.0 + MARGIN))
    
    points = points[mask]
    
    return points, mask




def one_hot(x, num_class=1):
    if num_class is None:
        num_class = 1
    ohx = np.zeros((len(x), num_class))
    ohx[range(len(x)), x] = 1
    return ohx


def global_alignment(points, axis_align_matrix, rotation_axis=2):
    def _trans_points(points, trans_factor):
        points[:, :3] += trans_factor
        return points

    def _rot_points(points, rot_mat):
        points[:, :3] = points[:, :3] @ rot_mat.T
        return points

    def _check_rot_mat(rot_mat, rotation_axis=2):
        is_valid = np.allclose(np.linalg.det(rot_mat), 1.0)
        valid_array = np.zeros(3)
        valid_array[rotation_axis] = 1.0
        is_valid &= (rot_mat[rotation_axis, :] == valid_array).all()
        is_valid &= (rot_mat[:, rotation_axis] == valid_array).all()
        assert is_valid, f'invalid rotation matrix {rot_mat}'

    rot_mat = axis_align_matrix[:3, :3]
    trans_vec = axis_align_matrix[:3, -1]
    _check_rot_mat(rot_mat, rotation_axis)
    points = _rot_points(points, rot_mat)
    points = _trans_points(points, trans_vec)

    return points

def point_seg_class_mapping(semantic_mask, valid_cat_ids, max_cat_id):
    assert max_cat_id >= np.max(np.array(valid_cat_ids)), \
        'max_cat_id should be greater than maximum id in valid_cat_ids'
    max_cat_id = int(max_cat_id)

    # build cat_id to class index mapping
    neg_cls = len(valid_cat_ids)
    cat_id2class = np.ones(max_cat_id + 1, dtype=np.int) * neg_cls
    for cls_idx, cat_id in enumerate(valid_cat_ids):
        cat_id2class[cat_id] = cls_idx

    converted_sem_mask = cat_id2class[semantic_mask]
    return converted_sem_mask

def points_random_sampling(points, num_samples, replace=None, return_choices=False):
    if replace is None:
        replace = (points.shape[0] < num_samples)
    choices = np.random.choice(
        points.shape[0], num_samples, replace=replace)
    if return_choices:
        return points[choices], choices
    else:
        return points[choices]

