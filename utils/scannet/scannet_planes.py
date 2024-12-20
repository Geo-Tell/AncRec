import json
import os
from time import time
import numpy as np
import torch
from utils import pc_util
from utils.scannet import scannet_utils
from configs.path_config import scans_dir, layout_path
SCAN_DIR = scans_dir
# sudo mount 192.168.210.155:/data/scannet/scans /home/dmy/data/scans
# sudo mount 192.168.210.155:/data/scannet/scannet_planes /home/dmy/data/scannet_planes
# sudo mount 192.168.210.155:/data/scannet/scan2cad /home/dmy/data/scan2cad
# sudo mount 192.168.210.155:/data/scannet/ShapeNetCore.v2 /home/dmy/data/ShapeNetCore.v2

# sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/scannet /home/dmy/data/datasets/scannet
# sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/scannet_planes /home/dmy/data/datasets/scannet_planes
# sudo mount 192.168.210.155:/data/scannet/scan2cad /home/dmy/data/scan2cad
# sudo mount 192.168.210.155:/data/scannet/scans /home/dmy/data/scans
# sudo mount 192.168.210.155:/data/scannet/scans /home/dmy/data/scans
# sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/ShapeNetv2_data /home/dmy/data/datasets/ShapeNetv2_data


# sudo mount 192.168.210.155:/data/scannet/dmy/datasets/processed_data /home/dmy/data/datasets/scannet/processed_data
# sudo mount 192.168.210.155:/data/scannet/dmy/datasets/ShapeNetv2_data/pointcloud /home/dmy/data/datasets/ShapeNetv2_data/pointcloud

# sudo mount 192.168.210.155:/data/scannet/dmy/datasets/posed_images_final /home/dmy/data/datasets/scannet/posed_images_final
# sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/posed_images_final /home/dmy/data/datasets/scannet/posed_images_final
# sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/posed_images_final_instance /home/dmy/data/datasets/scannet/posed_images_final_instance
# sudo mount 192.168.210.155:/data/scannet/dmy/datasets /home/dmy/data/datasets/scannet
# sudo mount 192.168.210.155:/data/scannet/scannet_planes /home/dmy/data/datasets/scannet_planes
# sudo mount 192.168.210.155:/data/scannet/scans /home/dmy/data/scans



def isFourPointsInSamePlane(p0, p1, p2, p3, error):
    s1 = p1 - p0
    s2 = p2 - p0
    s3 = p3 - p0
    result = s1[0] * s2[1] * s3[2] + s1[1] * s2[2] * s3[0] + s1[2] * s2[0] * s3[1] - s1[2] * s2[1] * s3[0] - s1[0] * s2[
        2] * s3[1] - s1[1] * s2[0] * s3[2]
    if result - error <= 0 <= result + error:
        return True
    return False

def get_normal(quad_vert, center):
    tmp_A = []
    tmp_b = []
    for i in range(4):
        tmp_A.append([quad_vert[i][0], quad_vert[i][1], 1])  # x,y,1
        tmp_b.append(quad_vert[i][2])  # z
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    temp = A.T * A
    if np.linalg.det(temp) > 1e-10:
        fit = np.array(temp.I * A.T * b)
        a = fit[0][0] / fit[2][0]
        b = fit[1][0] / fit[2][0]
        c = -1.0 / fit[2][0]
        normal_vector = np.array([a, b, c])

        # print ("solution:%f x + %f y + %f z + 1 = 0" % (a, b, c) )

    else:  # vertical
        b = np.matrix([-1, -1, -1, -1]).T
        A = A[:, 0:2]
        temp = A.T * A
        fit = np.array(temp.I * A.T * b)
        a = fit[0][0]
        b = fit[1][0]
        c = 0
        normal_vector = np.array([a, b, c])
        # print ("solution:%f x + %f y + 1 = 0" % (a, b) )

    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector

def rotate_quad(rectangle, rot_mat):
    centers = rectangle[:, 0:3]
    new_centers = np.dot(centers, np.transpose(rot_mat))
    normal_vector = rectangle[:, 3:6]
    new_normal_vector = np.dot(normal_vector, np.transpose(rot_mat))

    return np.concatenate([new_centers, new_normal_vector, rectangle[:, 6:8]], axis=1)

def projection2d(point,center,normal_vector,size):
    #中间的墙的问题
    point = point.double()
    a = normal_vector[:,0] #M
    b = normal_vector[:,1]
    d = -(a*center[:,0]+b*center[:,1]) #M,1
    # quad = torch.cat((a.view([1]),b.view([1])))
    quad = torch.cat((a.unsqueeze(0),b.unsqueeze(0)),dim=0) # 2,M
    delta = point[:,0:2].matmul(quad)+ d.unsqueeze(0).repeat(point.shape[0],1)
    # a = a.unsqueeze(0).repeat(point.shape[0],1)
    # b = b.unsqueeze(0).repeat(point.shape[0],1)
    # point = point.unsqueeze(1).repeat(1,quad.shape[1],1)
    # k = -(a * point[:,:, 0]+ b * point[:,:, 1]
    #       + d.unsqueeze(0).repeat(point.shape[0],1)) #N.M
    # x = point[:,:, 0]+a*k
    # y = point[:,:, 0]+b*k
    # t = torch.cat((x.unsqueeze(2),y.unsqueeze(2)),dim=2) #torch.Size([81369, 5, 2])
    # # w = torch.norm(t-center[:,0:2].unsqueeze(0).repeat(point.shape[0],1,1),dim=2)
    # w = torch.abs(t-center[:,0:2].unsqueeze(0).repeat(point.shape[0],1,1))
    # # w = torch.abs(t[:,:,0]-center[:,0].unsqueeze(0).repeat(point.shape[0],1))
    # # ind = torch.where(w > size[:,0])
    # ind = torch.where(w > size)
    # # delta[ind] = 10.0
    # delta[ind[0], ind[1]] = 10.0
    point = point[:,0:2].unsqueeze(1).repeat(1, quad.shape[1], 1) #N,M,2
    distc_ = torch.norm(point - center[:,0:2].unsqueeze(0).repeat(point.shape[0],1,1),dim=2)#N,M
    distc = torch.square(distc_) - torch.square(delta)
    ind = torch.where(distc > torch.square(size[:,0]/2))
    delta[ind] = 1.0
    return torch.abs(delta)

def rectangle(quad_vert, center):
    """
    input: p1,p2,p3,p4
    return: normal vector, size, quad center, direction
    """

    quad_center = np.mean(quad_vert, axis=0)

    normal_vector = get_normal(quad_vert, center)

    vertical_normal_vector = np.array([normal_vector[0], normal_vector[1], 0])

    # vertical_normal_vector = vertical_normal_vector / np.linalg.norm(vertical_normal_vector)

    edge_vector = quad_vert[0] - quad_vert[1]

    cos_theta = torch.cosine_similarity(torch.tensor(edge_vector), torch.tensor([0, 0, 1]), dim=0)

    l1 = np.linalg.norm(quad_vert[0] - quad_vert[1])
    l2 = np.linalg.norm(quad_vert[1] - quad_vert[2])
    l3 = np.linalg.norm(quad_vert[2] - quad_vert[3])
    l4 = np.linalg.norm(quad_vert[3] - quad_vert[0])
    l5 = (l1 + l3) / 2
    l6 = (l2 + l4) / 2

    if abs(cos_theta) > 0.5:
        h = np.array([l5])
        w = np.array([l6])
    else:
        h = np.array([l6])
        w = np.array([l5])

    rectangle = np.concatenate((quad_center, vertical_normal_vector, w, h))  # 3+3+2=8

    return rectangle

def get_center(verts):
    verts = np.array(verts)
    center = np.mean(verts, axis=0)
    return center

def transform(scene_name, mesh_vertices):
    # meta_file = os.path.join(path_config.metadata_root, 'scans', scene_name, scene_name + '.txt')  # includes axis
    meta_file = os.path.join(SCAN_DIR, scene_name, scene_name + '.txt')  # includes axis
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                                 for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]
    return mesh_vertices

def get_quads(scan_name):
    with open(layout_path + '/' + scan_name + '.json', 'r') as quad_file:
        plane_dict = json.load(quad_file)
    quad_dict = plane_dict['quads']
    total_quad_num = len(quad_dict)

    vert_dict = plane_dict['verts']

    for i in range(0, len(vert_dict)):
        temp = vert_dict[i][1]
        vert_dict[i][1] = - vert_dict[i][2]
        vert_dict[i][2] = temp

    verts = np.array(vert_dict)

    verts = transform(scan_name, verts)

    quads = [i for i in quad_dict if len(i) == 4]

    quad_verts = np.asarray([[verts[j] for j in _] for _ in quads])  # 获取每个quad的顶点

    # 验证共面性
    quad_verts_filter_ = np.asarray([quad_vert for quad_vert in quad_verts
                                     if isFourPointsInSamePlane(quad_vert[0], quad_vert[1], quad_vert[2], quad_vert[3],
                                                                100)])

    room_center = get_center(vert_dict)  # room center

    quad_verts_filter = np.asarray([quad_vert for quad_vert in quad_verts_filter_
                                    if abs(get_normal(quad_vert, room_center)[2]) < 0.2])  # only vertical

    horizontal_quads = np.asarray([quad_vert for quad_vert in quad_verts_filter_
                                   if abs(get_normal(quad_vert, room_center)[2]) > 0.8])  # only horizontal

    rectangles = np.array([rectangle(_, room_center) for _ in quad_verts_filter])

    return rectangles, total_quad_num, horizontal_quads

def vis_quad(scene_name, rectangle, output_pth = 'test_svote/'):
    obbs = []
    for i in range(rectangle.shape[0]):
        #quad_center, vertical_normal_vector, w, h)
        vector_label = rectangle[i,3:6]
        cos_theta = torch.cosine_similarity(torch.tensor(vector_label), torch.tensor([0, 1, 0]), dim=0)
        heading_angle = torch.arccos(cos_theta)
        cos_theta1 = torch.cosine_similarity(torch.tensor(vector_label), torch.tensor([1, 0, 0]), dim=0)
        if cos_theta1 > 0:
            heading_angle = np.pi * 2 - heading_angle
        obb = np.zeros((7,))
        obb[0:3] = rectangle[i,0:3]
        obb[3] =  rectangle[i,6]
        obb[4] = 0.1
        obb[5] = rectangle[i,7]
        obb[6] = heading_angle
        obbs.append(obb)

    if len(obbs) > 0:
        obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
        pc_util.write_oriented_bbox(obbs,  output_pth + '%s_gt_quad.ply' % (scene_name))



