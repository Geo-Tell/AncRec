import open3d as o3d
import os
import re
import numpy as np
import argparse
from configs.path_config import scannet_processed_path
ScanNet_OBJ_CLASS_NAMES =  ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']
palette_inst = np.array([
    [0,245,255], #blue 0 table
    [127,255,212],#Aquamarine1 (blue-green) 1 , chair
    [255,215,0], #gold 2 , bookshelf
    [255,130,171],#paleVioletdRed2 3, sofa
    [255,187,255],#plum1 (purple) 4 , trash_bin
    [255,99,71], #tomato 5, cabinet
    [255,165,0], #orange , display
    [192,255,62], #olivedrab1 (light green) bathtub
    [255,228,181] # moccasin (light orange)
])

def run_vis(scene_name, mesh_path, pc_path,layout_path=None):
    instances = []
    # vis.draw(pcd)
    input_point_cloud = np.load(os.path.join(pc_path, scene_name + '.npy'))[:, :3]
    pc_center = np.mean(input_point_cloud, axis=0)
    pcd = o3d.geometry.PointCloud()
    # vis.get_render_option().point_size = 1
    pcd.points = o3d.utility.Vector3dVector(input_point_cloud)
    pcd.paint_uniform_color([0.2, 0.2, 0.2])
    instances.append({"name": "pc", "geometry": pcd})

    # scene_name_for_mesh = '_'.join(scene_name.split('_')[1:])
    mesh_file_in_scene = [f for f in os.listdir(mesh_path) if re.search(r'%s.*.ply' % scene_name, f)]
    if len(mesh_file_in_scene) == 0:
        print('scene_name: %s: no result!' % scene_name)
        return

    for mesh_file in mesh_file_in_scene:
        if len(mesh_file.split('_')) < 4:
            continue
        cls_name = mesh_file.split('_')[3]
        cls_name = cls_name.split('.')[0]
        if mesh_path.split('/')[-1] == "rfd" or mesh_path.split('/')[
            -1] == "dimr":  # mesh_path.split('/')[-1] == "dimr"or
            obj_prob = float(mesh_file.split('_')[-1].rstrip('.ply'))
            cls_prob = 1
        elif mesh_path.split('/')[-1] == "gt_meshes":  # "recon":
            obj_prob = 1
            cls_prob = 1
        else:
            obj_prob = float(mesh_file.split('_')[-2])
            cls_prob = float(mesh_file.split('_')[-1].rstrip('.ply'))

        if obj_prob < 0.3 or cls_prob < 0.4:
            continue
        cls_id = ScanNet_OBJ_CLASS_NAMES.index(cls_name)
        color = palette_inst[int(cls_id)].astype(float) / 255.0

        mesh_file_path = os.path.join(mesh_path, mesh_file)
        mesh = o3d.io.read_triangle_mesh(mesh_file_path)
        if mesh_path.split('\\')[-1] == 'ours2':
            R = mesh.get_rotation_matrix_from_quaternion(np.array([0, 0, 0, -1]))
            # print(pc_center)
            mesh.rotate(R, (pc_center[0], pc_center[1], pc_center[2]))
            # mesh.translate((-pc_center[0],-pc_center[1],-pc_center[2]))

        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        # vis.add_geometry(mesh)
        instances.append({"name": "mesh_%s" % mesh_file.split('.')[0], "geometry": mesh})

    if layout_path is not None:
        layout_file_in_scene = [f for f in os.listdir(layout_path) if '_'.join(f.split('_')[:2]) == scene_name]
        # layout_file_in_scene = [f for f in os.listdir(layout_path) if '_'.join(f.split('_')[1:3])==scene_name]
        if len(layout_file_in_scene) == 1:
            # vis.get_render_option().line_width = 10
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = 5  # note that this is scaled with respect to pixels,

            layout_file = np.load(os.path.join(layout_path, layout_file_in_scene[0]))
            if mesh_path.split('/')[-1] == 'gt_meshes':
                floors = layout_file['gt_floors']
                wall_height = layout_file['gt_ceilings'][:, 2].mean() - floors[:, 2].mean()
            else:
                # if mesh_path.split('/')[-1] == 'ours2':
                floors = layout_file['floors']
                wall_height = layout_file['wall_height']

            # floors = np.vstack([floors, floors[0,:]])
            ceilings = floors.copy()
            ceilings[:, 2] = ceilings[:, 2] + wall_height
            lines = []
            for vid in range(len(floors)):
                if vid == len(floors) - 1:
                    lines.append([vid, 0])
                else:
                    lines.append([vid, vid + 1])

            colors = [[0, 0, 0] for i in range(len(lines))]
            line_set_floor = o3d.geometry.LineSet()
            line_set_floor.lines = o3d.utility.Vector2iVector(lines)
            line_set_floor.colors = o3d.utility.Vector3dVector(colors)
            line_set_floor.points = o3d.utility.Vector3dVector(floors)
            # line_set_floor.material = mat
            # vis.add_geometry(geometry=line_set_floor, material = mat)
            # vis.draw(line_set_floor)
            instances.append({"name": "floors", "geometry": line_set_floor, "material": mat})

            line_set_ceilings = o3d.geometry.LineSet()
            line_set_ceilings.lines = o3d.utility.Vector2iVector(lines)
            line_set_ceilings.colors = o3d.utility.Vector3dVector(colors)
            line_set_ceilings.points = o3d.utility.Vector3dVector(ceilings)
            # vis.add_geometry(line_set_ceilings)
            instances.append({"name": "ceilings", "geometry": line_set_ceilings, "material": mat})

            all_points = np.vstack([floors, ceilings])
            lines2 = []
            for vid in range(len(floors)):
                lines2.append([vid, vid + len(floors)])

            line_set_connection = o3d.geometry.LineSet()
            line_set_connection.lines = o3d.utility.Vector2iVector(lines2)
            line_set_connection.colors = o3d.utility.Vector3dVector(colors)
            line_set_connection.points = o3d.utility.Vector3dVector(all_points)
            # vis.add_geometry(line_set_connection)
            instances.append({"name": "connection", "geometry": line_set_connection, "material": mat})

    o3d.visualization.draw(instances)

def extract_pc(pc_path):
    for scene_name in os.listdir(scannet_processed_path):
        scan_data = np.load(os.path.join(scannet_processed_path,scene_name, "full_scan.npz"))
        point_cloud = scan_data['mesh_vertices'][:, :3]
        np.save(os.path.join(pc_path,"%s.npy"%scene_name), point_cloud)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--scene_names', nargs='+', help='visualize scene names', required=True)
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--pc_path', type=str, required=True)
    parser.add_argument('--layout_path', type=str, default='')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.pc_path):
        os.makedirs(args.pc_path)
        extract_pc(args.pc_path)

    # scene_names = ["scene0050_01", #"scene0050_00",
    #                "scene0378_02", #"scene0378_00",
    #                "scene0474_01", "scene0474_02",
    #                "scene0608_01",#"scene0608_00","scene0494_00",
    #                "scene0616_01","scene0690_00"]
    for scene_name in args.scene_names:
        if not os.path.exists(args.layout_path):
            args.layout_path = None
        run_vis(scene_name, mesh_path = args.mesh_path, pc_path =  args.pc_path, layout_path= args.layout_path)  #
