import os
import torch
import json
import torchvision
import trimesh
import argparse

from scipy.spatial import cKDTree
from tqdm import tqdm
import numpy as np
import open3d as o3d

from gaustudio.utils.sh_utils import SH2RGB
from gaustudio.utils.misc import load_config
from gaustudio import models, renderers
from gaustudio.datasets.utils import JSON_to_camera, getNerfppNorm

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def create_point_cloud(surface_xyz_np, surface_color_np, surface_normal_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_xyz_np)
    pcd.colors = o3d.utility.Vector3dVector(surface_color_np)
    pcd.normals = o3d.utility.Vector3dVector(surface_normal_np)
    return pcd

def remove_normal_outliers(pcd, nb_neighbors=20, angle_threshold=np.pi/4):
    normals = np.asarray(pcd.normals)
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    inlier_indices = []
    
    for i in range(len(pcd.points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], nb_neighbors)
        neighbor_normals = normals[idx[1:]]  # Exclude the point itself
        angles = np.arccos(np.abs(np.dot(neighbor_normals, normals[i])))
        if np.mean(angles) < angle_threshold:
            inlier_indices.append(i)
    
    return pcd.select_by_index(inlier_indices)

def clean_point_cloud(pcd, nb_neighbors=50, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_clean = pcd.select_by_index(ind)
    
    pcd_clean = remove_normal_outliers(pcd_clean)
    
    return pcd_clean

def mesh_nksr(input_xyz, input_normal, voxel_size=0.008, detail_level=0):
    try:
        from nksr import Reconstructor, utils, fields
    except:
        raise ImportError("Please install nksr to use this feature.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_xyz = input_xyz.to(device)
    input_normal = input_normal.to(device)
    reconstructor = Reconstructor(device)
    field = reconstructor.reconstruct(input_xyz, input_normal, voxel_size=voxel_size, detail_level=detail_level)
    mesh = field.extract_dual_mesh(mise_iter=2)
    return trimesh.Trimesh(vertices=mesh.v.cpu().numpy(), faces=mesh.f.cpu().numpy())

def mesh_poisson(pcd, depth=8,  density_threshold=0.01):    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.2, linear_fit=False)

    densities = np.asarray(densities)
    densities = (densities - densities.min()) / (densities.max() - densities.min())

    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh

def mesh_pymeshlab_poisson(pcd_path, depth=8, smooth_iterations=3, smooth_lambda=0.5, clean_threshold=10):
    try:
        import pymeshlab
    except ImportError:
        raise ImportError("Please install pymeshlab to use this feature.")

    # Create a new MeshSet
    ms = pymeshlab.MeshSet()

    # Load the point cloud directly using pymeshlab
    ms.load_new_mesh(pcd_path)

    # Apply Poisson surface reconstruction with tunable parameters
    ms.apply_filter('generate_surface_reconstruction_screened_poisson', 
                    depth=depth)

    # Smooth the mesh using Laplacian smoothing
    ms.apply_filter('apply_coord_laplacian_smoothing', 
                    stepsmoothnum=smooth_iterations)

    # Get the final reconstructed mesh
    reconstructed_mesh = ms.current_mesh()

    # Convert pymeshlab mesh to trimesh
    vertices = reconstructed_mesh.vertex_matrix()
    faces = reconstructed_mesh.face_matrix()
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def mesh_sap(pcd):
    from gaustudio.models import ShapeAsPoints
    sap_pcd = ShapeAsPoints.from_o3d_pointcloud(pcd)
    mesh = sap_pcd.to_o3d_mesh()
    return mesh

def normal_fusion(pcd, all_ids_list, all_normals_list, all_confidences_list, cameras):
    device = pcd._xyz.device
    unique_ids, inverse_indices = torch.unique(torch.cat(all_ids_list), return_inverse=True)
    num_unique_ids = len(unique_ids)
    
    sum_normals = torch.zeros((num_unique_ids, 3), device=device)
    sum_weights = torch.zeros(num_unique_ids, device=device)
    
    start_idx = 0
    for ids, normals, confidences, camera in zip(all_ids_list, all_normals_list, all_confidences_list, cameras):
        # Compute view-dependent weights
        view_dir = camera.extrinsics[:3, 3].to(device) - pcd._xyz[ids]
        view_dir = view_dir / torch.norm(view_dir, dim=1, keepdim=True)
        view_weights = torch.abs(torch.sum(view_dir * normals, dim=1))
        
        distances = torch.norm(camera.extrinsics[:3, 3].to(device) - pcd._xyz[ids], dim=1)
        distance_weights = 1 / (distances + 1e-6)
        
        # Combine weights
        weights = confidences * view_weights * distance_weights
        
        # Accumulate weighted normals
        end_idx = start_idx + len(ids)
        sum_normals.index_add_(0, inverse_indices[start_idx:end_idx], normals * weights.unsqueeze(1))
        sum_weights.index_add_(0, inverse_indices[start_idx:end_idx], weights)
        
        start_idx = end_idx
    
    # Compute mean normals
    mean_normals = sum_normals / sum_weights.unsqueeze(1)
    mean_normals = torch.nn.functional.normalize(mean_normals, p=2, dim=1)
    
    # Consistency checking and recomputation
    start_idx = 0
    sum_normals.zero_()
    sum_weights.zero_()
    for ids, normals, confidences, camera in zip(all_ids_list, all_normals_list, all_confidences_list, cameras):
        view_dir = camera.extrinsics[:3, 3].to(device) - pcd._xyz[ids]
        view_dir = view_dir / torch.norm(view_dir, dim=1, keepdim=True)
        view_weights = torch.abs(torch.sum(view_dir * normals, dim=1))
        
        distances = torch.norm(camera.extrinsics[:3, 3].to(device) - pcd._xyz[ids], dim=1)
        distance_weights = 1 / (distances + 1e-6)
        
        weights = confidences * view_weights * distance_weights
        
        end_idx = start_idx + len(ids)
        current_inverse_indices = inverse_indices[start_idx:end_idx]
        
        normal_diff = torch.norm(normals - mean_normals[current_inverse_indices], dim=1)
        consistency_mask = normal_diff < 0.8  # Adjust threshold as needed
        
        sum_normals.index_add_(0, current_inverse_indices[consistency_mask], 
                               normals[consistency_mask] * weights[consistency_mask].unsqueeze(1))
        sum_weights.index_add_(0, current_inverse_indices[consistency_mask], weights[consistency_mask])
        
        start_idx = end_idx
    
    mean_normals = sum_normals / sum_weights.unsqueeze(1)
    mean_normals = torch.nn.functional.normalize(mean_normals, p=2, dim=1)
    
    # Spatial smoothing (you might want to move this to GPU for large point clouds)
    surface_xyz = pcd._xyz[unique_ids].cpu().numpy()
    tree = cKDTree(surface_xyz)
    k = 10  # number of neighbors
    distances, indices = tree.query(surface_xyz, k=k)
    
    smoothed_normals = torch.zeros_like(mean_normals)
    for i in range(num_unique_ids):
        neighbor_normals = mean_normals[indices[i]]
        spatial_weights = torch.exp(-torch.tensor(distances[i], device=device) / 0.1)
        normal_diff = torch.norm(neighbor_normals - mean_normals[i], dim=1)
        normal_weights = torch.exp(-normal_diff / 0.5)
        smooth_weights = spatial_weights * normal_weights
        smooth_weights = smooth_weights / smooth_weights.sum()
        smoothed_normals[i] = torch.sum(neighbor_normals * smooth_weights.unsqueeze(1), dim=0)
    
    smoothed_normals = torch.nn.functional.normalize(smoothed_normals, p=2, dim=1)
    
    return unique_ids, smoothed_normals

def clean_mesh(input_cloud_path, mesh_path, relative_distance_threshold=0.01):
    # Load input point cloud
    input_cloud = o3d.io.read_point_cloud(input_cloud_path)
    input_points = np.asarray(input_cloud.points)

    # Load input mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Get mesh vertices
    mesh_vertices = np.asarray(mesh.vertices)

    # Calculate mesh scale (using bounding box diagonal)
    bbox = mesh.get_axis_aligned_bounding_box()
    mesh_scale = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())

    # Calculate absolute distance threshold based on mesh scale
    distance_threshold = relative_distance_threshold * mesh_scale

    # Create KD-tree from input points
    kdtree = cKDTree(input_points)

    # Find nearest neighbors for mesh vertices
    distances, _ = kdtree.query(mesh_vertices)

    # Identify outliers
    is_outlier = distances > distance_threshold

    # Create clean vertices
    clean_vertices = mesh_vertices[~is_outlier]

    # Create mapping from old indices to new indices
    old_to_new = np.cumsum(~is_outlier) - 1
    old_to_new[is_outlier] = -1

    # Update triangle indices
    triangles = np.asarray(mesh.triangles)
    valid_triangles = np.all(old_to_new[triangles] != -1, axis=1)
    clean_triangles = old_to_new[triangles[valid_triangles]]

    # Create clean mesh
    clean_mesh = o3d.geometry.TriangleMesh()
    clean_mesh.vertices = o3d.utility.Vector3dVector(clean_vertices)
    clean_mesh.triangles = o3d.utility.Vector3iVector(clean_triangles)

    # Remove degenerate triangles
    clean_mesh.remove_degenerate_triangles()
    clean_mesh.remove_unreferenced_vertices()

    return clean_mesh
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default='vanilla')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--camera', '-c', default=None, help='path to cameras.json')
    parser.add_argument('--model', '-m', default=None, help='path to the model')
    parser.add_argument('--output-dir', '-o', default=None, help='path to the output dir')
    parser.add_argument('--load_iteration', default=-1, type=int, help='iteration to be rendered')
    parser.add_argument('--resolution', default=1, type=int, help='downscale resolution')
    parser.add_argument('--sh', default=0, type=int, help='default SH degree')
    parser.add_argument('--white_background', action='store_true', help='use white background')
    parser.add_argument('--clean', action='store_true', help='perform a clean operation')
    parser.add_argument('--meshing', choices=['nksr', 'poisson', 'sap', \
                                              'poisson-9', 'pymeshlab-poisson', None], 
                        default='nksr', help='Meshing method to use')
    args, extras = parser.parse_known_args()
    
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))
    
    # parse YAML config to OmegaConf
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '../configs', args.config+'.yaml')
    config = load_config(config_path, cli_args=extras)
    config.cmd_args = vars(args)  
    
    pcd = models.make(config.model.pointcloud)
    renderer = renderers.make(config.renderer)
    pcd.active_sh_degree = args.sh
    
    model_path = args.model
    if os.path.isdir(model_path):
        if args.load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(args.model, "point_cloud"))
        else:
            loaded_iter = args.load_iteration
        work_dir = os.path.join(model_path, "renders", "iteration_{}".format(loaded_iter)) if args.output_dir is None else args.output_dir
        
        print("Loading trained model at iteration {}".format(loaded_iter))
        pcd.load(os.path.join(args.model,"point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))
        output_pcd_path = os.path.join(work_dir, "fused.ply")
    elif model_path.endswith(".ply"):
        work_dir = os.path.join(os.path.dirname(model_path), os.path.basename(model_path)[:-4]) if args.output_dir is None else args.output_dir
        pcd.load(model_path)
        output_pcd_path = model_path[:-4] + "_fused.ply"
    else:
        print("Model not found at {}".format(model_path))
    pcd.to("cuda")
    
    if args.camera is None:
        args.camera = os.path.join(model_path, "cameras.json")
    if os.path.exists(args.camera):
        print("Loading camera data from {}".format(args.camera))
        with open(args.camera, 'r') as f:
            camera_data = json.load(f)
        cameras = []
        for camera_json in camera_data:
            camera = JSON_to_camera(camera_json, "cuda")
            cameras.append(camera)
    else:
        assert "Camera data not found at {}".format(args.camera)
    
    render_path = os.path.join(work_dir, "images")
    normal_path = os.path.join(work_dir, "normals")
    mask_path = os.path.join(work_dir, "masks")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(normal_path, exist_ok=True)

    scene_radius = getNerfppNorm(cameras)["radius"]
    all_ids = []
    all_confidances = []
    all_normals = []

    for camera in tqdm(cameras):
        camera.downsample_scale(args.resolution)
        camera = camera.to("cuda")
        with torch.no_grad():
            render_pkg = renderer.render(camera, pcd)
        
        rendering = render_pkg["render"]
        rendered_final_opacity = render_pkg["rendered_final_opacity"][0] 
        rendered_depth = render_pkg["rendered_depth"][0]
        cam_normals = camera.depth2normal(rendered_depth, coordinate='camera')
        normals = camera.normal2worldnormal(cam_normals)
        median_point_depths = render_pkg["rendered_median_depth"][0]
        median_point_ids = render_pkg["rendered_median_id"][0]
        
        fg_mask = rendered_final_opacity > 0.1
        valid_mask = (median_point_depths < scene_radius * 0.8) & (rendered_final_opacity > 0.95)
        valid_mask = (normals.sum(dim=-1) > -3) & valid_mask

        median_point_ids = median_point_ids[valid_mask]
        median_point_normals = -normals[valid_mask]
        median_point_confidances = rendered_final_opacity[valid_mask]
        
        all_confidances.append(median_point_confidances)
        all_ids.append(median_point_ids)
        all_normals.append(median_point_normals)

        # torchvision.utils.save_image(rendering, os.path.join(render_path, f"{camera.image_name}.png"))
        # torchvision.utils.save_image((cam_normals.permute(2, 0, 1) + 1)/2, os.path.join(normal_path, f"{camera.image_name}.png"))
        # torchvision.utils.save_image(fg_mask.float(), os.path.join(mask_path, f"{camera.image_name}.png"))

        # Save camera information
        cam_path = os.path.join(render_path, f"{camera.image_name}.cam")
        K = camera.intrinsics.cpu().numpy()
        fx, fy = K[0, 0], K[1, 1]
        paspect = fy / fx
        width, height = camera.image_width, camera.image_height
        dim_aspect = width / height
        img_aspect = dim_aspect * paspect
        flen = fy / height if img_aspect < 1.0 else fx / width
        ppx, ppy = K[0, 2] / width, K[1, 2] / height

        P = camera.extrinsics.cpu().numpy()
        with open(cam_path, 'w') as f:
            s1, s2 = '', ''
            for i in range(3):
                for j in range(3):
                    s1 += str(P[i][j]) + ' '
                s2 += str(P[i][3]) + ' '
            f.write(s2 + s1[:-1] + '\n')
            f.write(f"{flen} 0 0 {paspect} {ppx} {ppy}\n")

    # fusion
    unique_ids, fused_normals = normal_fusion(pcd, all_ids, all_normals, all_confidances, cameras)

    surface_xyz = pcd._xyz[unique_ids]
    surface_color = SH2RGB(pcd._f_dc[unique_ids]).clip(0,1)
    surface_normal = fused_normals
    
    surface_xyz_np = surface_xyz.cpu().numpy()
    surface_color_np = surface_color.cpu().numpy()
    surface_normal_np = surface_normal.cpu().numpy()
    pcd = create_point_cloud(surface_xyz_np, surface_color_np, surface_normal_np)
    
    pcd = clean_point_cloud(pcd)
    print(f"Point cloud cleaned. Remaining points: {len(pcd.points)}")

    o3d.io.write_point_cloud(output_pcd_path, pcd)
    
    if args.meshing == 'nksr':
        input_xyz = torch.from_numpy(np.asarray(pcd.points)).float()
        input_normal = torch.from_numpy(np.asarray(pcd.normals)).float()
        mesh = mesh_nksr(input_xyz, input_normal)
        mesh.export(os.path.join(work_dir, "fused_mesh.ply"))
    elif args.meshing.startswith('poisson'):
        if args.meshing == 'poisson':
            depth = 8
        else:
            depth = int(args.meshing.split('-')[1])
        mesh = mesh_poisson(pcd, depth=depth)
        o3d.io.write_triangle_mesh(os.path.join(work_dir, f"fused_mesh.ply"), mesh)
    elif args.meshing.startswith('sap'):
        mesh = mesh_sap(pcd)
        o3d.io.write_triangle_mesh(os.path.join(work_dir, f"fused_mesh.ply"), mesh)
    elif args.meshing == 'pymeshlab-poisson':
        mesh = mesh_pymeshlab_poisson(output_pcd_path)
        mesh.export(os.path.join(work_dir, "fused_mesh.ply"))
        mesh = clean_mesh(output_pcd_path, os.path.join(work_dir, "fused_mesh.ply"))
        o3d.io.write_triangle_mesh(os.path.join(work_dir, f"fused_mesh.ply"), mesh)
    elif args.meshing == 'None':
        print("Skipping meshing as requested.")
    else:
        print(f"Unknown meshing method: {args.meshing}")

if __name__ == '__main__':
    main()