import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from bspt import get_mesh_watertight
import functools
from models.registers import MODULES
from plyfile import PlyData

def read_polymesh(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    faces = plydata['face'].data
    face_list = [ff[0] for ff in faces]
    return pc_array, face_list


class PolyMesh:
    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices)
        self.faces = faces
        self.tmesh = None

    def export(self, name):
        fout = open(name, 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex " + str(len(self.vertices)) + "\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("element face " + str(len(self.faces)) + "\n")
        fout.write("property list uchar int vertex_index\n")
        fout.write("end_header\n")
        for ii in range(len(self.vertices)):
            fout.write(
                str(self.vertices[ii][0]) + " " + str(self.vertices[ii][1]) + " " + str(self.vertices[ii][2]) + "\n")
        for ii in range(len(self.faces)):
            fout.write(str(len(self.faces[ii])))
            for jj in range(len(self.faces[ii])):
                fout.write(" " + str(self.faces[ii][jj]))
            fout.write("\n")
        fout.close()

    def sort_vertex_clockwise(self, vs):
        # vs: [N, 3], a list of coplanar 3D points.
        assert vs.shape[0] >= 3
        # calculate center
        c = vs.mean(0)
        # calculate plane normal
        n = np.cross(vs[0] - c, vs[1] - c)
        # argsort counterclockwise
        indices = sorted(range(vs.shape[0]),
                         key=functools.cmp_to_key(lambda i, j: np.dot(n, np.cross(vs[i] - c, vs[j] - c))))

        return indices

    def to_trimesh(self):
        # not lazy
        # if self.tmesh is None:
        if True:
            # triangulate polygons
            triangles = []
            # each face contains 3+ points
            for face in self.faces:
                if len(face) == 3:
                    triangles.append(face)
                else:
                    # split a 3d polygon into 3d triangles.
                    vs = []
                    for v in face:
                        vs.append(self.vertices[v])
                    vs = np.stack(vs, 0)
                    inds = self.sort_vertex_clockwise(vs)
                    for iind in range(1, len(face) - 1):
                        triangles.append([face[inds[0]], face[inds[iind]], face[inds[iind + 1]]])
            self.tmesh = trimesh.Trimesh(self.vertices, np.array(triangles), process=False)
        return self.tmesh

class generator(nn.Module):
    def __init__(self, phase, p_dim, c_dim):
        super(generator, self).__init__()
        self.phase = phase
        self.p_dim = p_dim
        self.c_dim = c_dim
        convex_layer_weights = torch.zeros((self.p_dim, self.c_dim))
        concave_layer_weights = torch.zeros((self.c_dim, 1))
        self.convex_layer_weights = nn.Parameter(convex_layer_weights)
        self.concave_layer_weights = nn.Parameter(concave_layer_weights)
        nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)
        nn.init.normal_(self.concave_layer_weights, mean=1e-5, std=0.02)

    def forward(self, points, plane_m, convex_mask=None):
        if self.phase == 0:
            # level 1
            h1 = torch.matmul(points, plane_m)
            h1 = torch.clamp(h1, min=0)

            # level 2
            h2 = torch.matmul(h1, self.convex_layer_weights)
            h2 = torch.clamp(1 - h2, min=0, max=1)

            # level 3
            h3 = torch.matmul(h2, self.concave_layer_weights)
            h3 = torch.clamp(h3, min=0, max=1)

            return h2, h3
        elif self.phase == 1 or self.phase == 2:
            # level 1
            h1 = torch.matmul(points, plane_m)
            h1 = torch.clamp(h1, min=0)

            # level 2
            h2 = torch.matmul(h1, (self.convex_layer_weights > 0.01).float())

            # level 3
            if convex_mask is None:
                h3 = torch.min(h2, dim=2, keepdim=True)[0]
            else:
                h3 = torch.min(h2 + convex_mask, dim=2, keepdim=True)[0]

            return h2, h3
        elif self.phase == 3 or self.phase == 4:
            # level 1
            h1 = torch.matmul(points, plane_m)
            h1 = torch.clamp(h1, min=0)

            # level 2
            h2 = torch.matmul(h1, self.convex_layer_weights)

            # level 3
            if convex_mask is None:
                h3 = torch.min(h2, dim=2, keepdim=True)[0]
            else:
                h3 = torch.min(h2 + convex_mask, dim=2, keepdim=True)[0]

            return h2, h3


class decoder(nn.Module):
    def __init__(self, ef_dim, p_dim):
        super(decoder, self).__init__()
        self.ef_dim = ef_dim
        self.p_dim = p_dim
        self.linear_1 = nn.Linear(self.ef_dim * 8, self.ef_dim * 16, bias=True)
        self.linear_2 = nn.Linear(self.ef_dim * 16, self.ef_dim * 32, bias=True)
        self.linear_3 = nn.Linear(self.ef_dim * 32, self.ef_dim * 64, bias=True)
        self.linear_4 = nn.Linear(self.ef_dim * 64, self.p_dim * 4, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias, 0)

    def forward(self, inputs):
        l1 = self.linear_1(inputs)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        l4 = self.linear_4(l3)
        l4 = l4.view(-1, 4, self.p_dim)

        return l4

class encoder(nn.Module):
    def __init__(self, ef_dim):
        super(encoder, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim * 4, self.ef_dim * 8, 4, stride=2, padding=1, bias=True)
        self.conv_5 = nn.Conv3d(self.ef_dim * 8, self.ef_dim * 8, 4, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.constant_(self.conv_1.bias, 0)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.constant_(self.conv_2.bias, 0)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.constant_(self.conv_3.bias, 0)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.constant_(self.conv_4.bias, 0)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)

    def forward(self, inputs):
        d_1 = self.conv_1(inputs) #1,1,64,64,64 -> 1,32(C),32,32,32
        d_1 = F.leaky_relu(d_1, negative_slope=0.01, inplace=True)

        d_2 = self.conv_2(d_1) #1,64,16,16,16
        d_2 = F.leaky_relu(d_2, negative_slope=0.01, inplace=True)

        d_3 = self.conv_3(d_2) #1,128,8,8,8
        d_3 = F.leaky_relu(d_3, negative_slope=0.01, inplace=True)

        d_4 = self.conv_4(d_3) #1,256,4,4,4
        d_4 = F.leaky_relu(d_4, negative_slope=0.01, inplace=True)

        d_5 = self.conv_5(d_4) #1,256,1,1,1
        d_5 = d_5.view(-1, self.ef_dim * 8)
        d_5 = torch.sigmoid(d_5)

        return d_5

@MODULES.register_module
class BSPNet(nn.Module):
    def __init__(self, config, optim_spec=None):
        super(BSPNet, self).__init__()
        self.optim_spec = optim_spec

        if 'sample_vox_size' in config:
            self.sample_vox_size = config['sample_vox_size']
        else:
            self.sample_vox_size = 64
        if self.sample_vox_size == 16:
            self.load_point_batch_size = 16 * 16 * 16
        elif self.sample_vox_size == 32:
            self.load_point_batch_size = 16 * 16 * 16
        elif self.sample_vox_size == 64:
            self.load_point_batch_size = 16 * 16 * 16 * 4
        self.shape_batch_size = 24
        self.point_batch_size = 16 * 16 * 16
        self.input_size = 64  # input voxel grid size

        self.real_size = 64  # output point-value voxel grid size in testing
        self.test_size = 32  # related to testing batch_size, adjust according to gpu memory size

        # get coords
        dima = self.test_size
        dim = self.real_size
        self.aux_x = np.zeros([dima, dima, dima], np.uint8)
        self.aux_y = np.zeros([dima, dima, dima], np.uint8)
        self.aux_z = np.zeros([dima, dima, dima], np.uint8)
        self.multiplier = int(dim / dima)



        self.ef_dim = 32
        self.p_dim = 4096
        self.c_dim = 256


        ''' build model'''
        self.decoder = decoder(self.ef_dim, self.p_dim)
        self.phase = 1
        self.generator = generator(self.phase, self.p_dim, self.c_dim)
        if self.phase == 0:
            # phase 0 continuous for better convergence
            # L_recon + L_W + L_T
            # G2 - network output (convex layer), the last dim is the number of convexes
            # G - network output (final output)
            # point_value - ground truth inside-outside value for each point
            # cw2 - connections T
            # cw3 - auxiliary weights W
            def network_loss(G2, G, point_value, cw2, cw3):
                loss_sp = torch.mean((point_value - G) ** 2)
                loss = loss_sp + torch.sum(torch.abs(cw3 - 1)) + (
                    torch.sum(torch.clamp(cw2 - 1, min=0) - torch.clamp(cw2, max=0)))
                return loss_sp, loss

            self.loss = network_loss
        elif self.phase == 1:
            # phase 1 hard discrete for bsp
            # L_recon
            def network_loss(G2, G, point_value, cw2, cw3):
                loss_sp = torch.mean(
                    (1 - point_value) * (1 - torch.clamp(G, max=1)) + point_value * (torch.clamp(G, min=0)))
                loss = loss_sp
                return loss_sp, loss

            self.loss = network_loss
        elif self.phase == 2:
            # phase 2 hard discrete for bsp with L_overlap
            # L_recon + L_overlap
            def network_loss(G2, G, point_value, cw2, cw3):
                loss_sp = torch.mean(
                    (1 - point_value) * (1 - torch.clamp(G, max=1)) + point_value * (torch.clamp(G, min=0)))
                G2_inside = (G2 < 0.01).float()
                bmask = G2_inside * (torch.sum(G2_inside, dim=2, keepdim=True) > 1).float()
                loss = loss_sp - torch.mean(G2 * point_value * bmask)
                return loss_sp, loss

            self.loss = network_loss
        elif self.phase == 3:
            # phase 3 soft discrete for bsp
            # L_recon + L_T
            # soft cut with loss L_T: gradually move the values in T (cw2) to either 0 or 1
            def network_loss(G2, G, point_value, cw2, cw3):
                loss_sp = torch.mean(
                    (1 - point_value) * (1 - torch.clamp(G, max=1)) + point_value * (torch.clamp(G, min=0)))
                loss = loss_sp + torch.sum((cw2 < 0.01).float() * torch.abs(cw2)) + torch.sum(
                    (cw2 >= 0.01).float() * torch.abs(cw2 - 1))
                return loss_sp, loss

            self.loss = network_loss
        elif self.phase == 4:
            # phase 4 soft discrete for bsp with L_overlap
            # L_recon + L_T + L_overlap
            # soft cut with loss L_T: gradually move the values in T (cw2) to either 0 or 1
            def network_loss(G2, G, point_value, cw2, cw3):
                loss_sp = torch.mean(
                    (1 - point_value) * (1 - torch.clamp(G, max=1)) + point_value * (torch.clamp(G, min=0)))
                G2_inside = (G2 < 0.01).float()
                bmask = G2_inside * (torch.sum(G2_inside, dim=2, keepdim=True) > 1).float()
                loss = loss_sp + torch.sum((cw2 < 0.01).float() * torch.abs(cw2)) + torch.sum(
                    (cw2 >= 0.01).float() * torch.abs(cw2 - 1)) - torch.mean(G2 * point_value * bmask)
                return loss_sp, loss

            self.loss = network_loss

    def forward(self, data, latent_features):
        end_points = {}
        plane_m = self.decoder(latent_features)
        end_points['plane_ms'] = plane_m
        if not self.feature_matching:
            data_points = data['data_points']
            data_values = data['data_values']
            point_batch_num = int(self.load_point_batch_size / self.point_batch_size)
            if point_batch_num == 1:
                point_coord = data_points
                point_value =  data_values
            else:
                which_batch = np.random.randint(point_batch_num)
                point_coord = data_points[:, which_batch * self.point_batch_size:(which_batch + 1) * self.point_batch_size,:]
                point_value = data_values[:,which_batch * self.point_batch_size:(which_batch + 1) * self.point_batch_size,:]


            net_out_convexes, net_out = self.generator(point_coord, plane_m, convex_mask=None)

            errSP, errTT = self.loss(net_out_convexes, net_out, point_value,
                                     self.generator.convex_layer_weights,
                                     self.generator.concave_layer_weights)
            end_points['loss_TT'] = errTT
            if self.phase != 1:
                end_points['loss_SP'] = errSP
        return end_points

    # output bsp shape as ply and point cloud as ply
    def generate(self, latent_features, gt_z_vector = None): #(self, data, gt_z_plane = None)
        plane_m = self.decoder(latent_features)
        if gt_z_vector is not None:
            gt_plane_m = self.decoder(gt_z_vector)

        dima = self.test_size
        dim = self.real_size
        multiplier = int(dim / dima)
        end_points = {}
        meshes = []
        gt_meshes = []

        for batchid in range(latent_features.shape[0]):
            if gt_z_vector is not None:
                gt_mesh = self.generate_from_latent(gt_plane_m[batchid].unsqueeze(0),multiplier)
                gt_meshes.append(gt_mesh)
            mesh = self.generate_from_latent(plane_m[batchid].unsqueeze(0),multiplier)
            meshes.append(mesh)



        # end_points['pred_zs'] = pred_z
        # end_points['plane_ms'] = plane_m
        end_points['meshes'] = meshes
        if gt_z_vector is not  None:
            end_points['gt_meshes'] = gt_meshes
        return end_points

    def generate_from_latent(self, plane_m, multiplier):
        multiplier2 = multiplier * multiplier
        model_float = np.ones([self.real_size, self.real_size, self.real_size, self.c_dim], np.float32)
        model_float_combined = np.ones([self.real_size, self.real_size, self.real_size], np.float32)
        coords = self.prepare_coords()
        coords = coords.to(plane_m.device)

        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    minib = i * multiplier2 + j * multiplier + k
                    point_coord = coords[minib:minib + 1]
                    model_out, model_out_combined = self.generator(point_coord, plane_m)

                    model_float[self.aux_x + i, self.aux_y + j, self.aux_z + k, :] = np.reshape(
                        model_out.detach().cpu().numpy(),
                        [self.test_size, self.test_size, self.test_size, self.c_dim])
                    model_float_combined[self.aux_x + i, self.aux_y + j, self.aux_z + k] = np.reshape(
                        model_out_combined.detach().cpu().numpy(), [self.test_size, self.test_size, self.test_size])

        out_m_ = plane_m.detach().cpu().numpy()

        # whether to use post processing to remove convexes that are inside the shape
        # post_processing_flag = False

        bsp_convex_list = []
        model_float = model_float < 0.01
        w2 = self.generator.convex_layer_weights.detach().cpu().numpy()

        for i in range(self.c_dim):
            slice_i = model_float[:, :, :, i]
            if np.max(slice_i) > 0:  # if one voxel is inside a convex
                # if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
                #	model_float_sum = model_float_sum-slice_i
                # else:
                box = []
                for j in range(self.p_dim):
                    if w2[j, i] > 0.01:
                        a = -out_m_[0, 0, j]
                        b = -out_m_[0, 1, j]
                        c = -out_m_[0, 2, j]
                        d = -out_m_[0, 3, j]
                        box.append([a, b, c, d])
                if len(box) > 0:
                    bsp_convex_list.append(np.array(box, np.float32))

        # convert bspt to mesh
        # use the following alternative to merge nearby vertices to get watertight meshes
        vertices, polygons = get_mesh_watertight(bsp_convex_list)
        mesh = PolyMesh(vertices, polygons)
        return mesh

    def prepare_coords(self):
        test_point_batch_size = self.test_size * self.test_size * self.test_size  # do not change

        # get coords
        dima = self.test_size
        dim = self.real_size
        multiplier = int(dim / dima)
        multiplier2 = multiplier * multiplier
        multiplier3 = multiplier * multiplier * multiplier

        for i in range(dima):
            for j in range(dima):
                for k in range(dima):
                    self.aux_x[i, j, k] = i * multiplier
                    self.aux_y[i, j, k] = j * multiplier
                    self.aux_z[i, j, k] = k * multiplier
        coords = np.zeros([multiplier3, dima, dima, dima, 3],
                               np.float32)  # 把一个物体分成不同的batch,间隔着来（2，4，6,..64/ 1,3,5,...63)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    coords[i * multiplier2 + j * multiplier + k, :, :, :, 0] = self.aux_x + i
                    coords[i * multiplier2 + j * multiplier + k, :, :, :, 1] = self.aux_y + j
                    coords[i * multiplier2 + j * multiplier + k, :, :, :, 2] = self.aux_z + k
        coords = (coords + 0.5) / dim - 0.5
        coords = np.reshape(coords, [multiplier3, test_point_batch_size, 3])
        coords = np.concatenate([coords, np.ones([multiplier3, test_point_batch_size, 1], np.float32)],
                                     axis=2)
        coords = torch.from_numpy(coords)

        return coords