import click
from gaustudio import datasets, models
from gaustudio.pipelines import initializers
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim

def procrustes_analysis(A, B):
    if A.shape != B.shape:
        raise ValueError("Input tensors A and B must have the same shape")

    batch_size, num_points, dims = A.shape
    R_optimal_batch = np.zeros((batch_size, dims, dims))
    scale_batch = np.zeros(batch_size)
    centroid_A_batch = np.zeros((batch_size, dims))
    centroid_B_batch = np.zeros((batch_size, dims))
    
    for i in range(batch_size):
        A_i = A[i]
        B_i = B[i]

        # Translate the points to the origin
        centroid_A = np.mean(A_i, axis=0)
        centroid_B = np.mean(B_i, axis=0)
        A_i -= centroid_A
        B_i -= centroid_B

        # Scale the points so that the mean of A_i's and B_i's variance is 1
        norm_A = np.linalg.norm(A_i)
        norm_B = np.linalg.norm(B_i)
        A_i /= norm_A
        B_i /= norm_B

        # Compute the optimal rotation matrix using singular value decomposition (SVD)
        U, S, Vt = np.linalg.svd(A_i.T @ B_i)
        R_optimal = Vt.T @ U.T

        # Correct reflection issue in SVD
        if np.linalg.det(R_optimal) < 0:
            Vt[-1, :] *= -1
            R_optimal = Vt.T @ U.T

        scale = norm_B / norm_A

        R_optimal_batch[i] = R_optimal
        scale_batch[i] = scale
        centroid_A_batch[i] = centroid_A
        centroid_B_batch[i] = centroid_B
    
    return R_optimal_batch, scale_batch, centroid_A_batch, centroid_B_batch

def rotation_matrix_to_euler_angles(R):
    # Convert the rotation matrix to Euler angles (yaw, pitch, roll)
    r = R.from_matrix(R)
    return r.as_euler('zyx', degrees=False)  # Yaw, Pitch, Roll

def compute_transformation_params(A, B):
    # Initial Procrustes analysis to get a rough estimate of rotation and scale
    R_optimal, scale, centroid_A, centroid_B = procrustes_analysis(A, B)

    # Extract Euler angles from rotation matrix
    yaw, pitch, roll = rotation_matrix_to_euler_angles(R_optimal)

    # Compute initial translation
    translation = centroid_B - scale * R_optimal @ centroid_A

    return yaw, pitch, roll, scale, translation

def extrinsics_to_yaw_pitch_roll_xyz(extrinsic_matrix):
    """
    Convert an extrinsic matrix to yaw, pitch, roll and translation (x, y, z).
    
    Args:
        extrinsic_matrix (numpy.ndarray): A 4x4 transformation matrix.
        
    Returns:
        tuple: A tuple containing yaw, pitch, roll in radians and translation (x, y, z).
    """
    
    # Ensure the matrix is 4x4
    assert extrinsic_matrix.shape == (4, 4), "The extrinsic matrix must be 4x4"
    
    # Extract translation components
    x, y, z = extrinsic_matrix[0, 3], extrinsic_matrix[1, 3], extrinsic_matrix[2, 3]
    
    # Extract rotation components
    R = extrinsic_matrix[0:3, 0:3]
    
    # Calculate Euler angles
    # Yaw (Z), Pitch (Y), Roll (X)
    pitch = np.arcsin(-R[2, 0])
    
    if np.cos(pitch) != 0:
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    else:
        # Gimbal lock occurs, handle singularity
        roll = 0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    
    return yaw, pitch, roll, x, y, z
import torch
import torch.optim as optim

def quaternion_to_rotation_matrix(q):
    # Normalize quaternion
    q = q / torch.norm(q)
    w, x, y, z = q.unbind()
    
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w]),
        torch.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w]),
        torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y])
    ])
    
    return R

def optimize_pose(A, B, num_iterations=1000, rotation_lr=0.01, translation_lr=0.01, scale_lr=0.01):
    A = torch.tensor(A, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)

    # Initialize parameters
    euler_angles = torch.zeros(3, requires_grad=True)  # [yaw, pitch, roll]
    log_scale = torch.tensor([1.0], requires_grad=True)  # Initialize at log(1) = 0
    t = torch.zeros(3, requires_grad=True)

    # Define optimizers
    rot_optimizer = optim.Adam([euler_angles], lr=rotation_lr)
    
    def euler_to_rotation_matrix(euler_angles):
        yaw, pitch, roll = euler_angles
        cos, sin = torch.cos, torch.sin
        R = torch.stack([
            torch.stack([cos(yaw)*cos(pitch), cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll)]),
            torch.stack([sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll)]),
            torch.stack([-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)])
        ])
        return R

    def get_transformation_matrix(euler_angles, log_scale, t):
        R = euler_to_rotation_matrix(euler_angles)
        scale = torch.exp(log_scale)
        T = torch.eye(4)
        T[:3, :3] = scale * R
        T[:3, 3] = scale * t
        return T

    # Optimization loop
    for phase in ['rotation', 'scale_and_translation']:
        optimizer = rot_optimizer if phase == 'rotation' else optim.Adam([log_scale, t], lr=scale_lr)
        for i in range(num_iterations):
            optimizer.zero_grad()

            T = get_transformation_matrix(euler_angles, log_scale, t)
            A_prime = torch.bmm(A, T.unsqueeze(0).expand(A.shape[0], -1, -1))

            if phase == 'rotation':
                loss = torch.mean((A_prime[:, :3, :3] - B[:, :3, :3]) ** 2)
            else:
                loss = torch.mean((A_prime - B) ** 2)

            if i % 100 == 0:
                print(f'{phase.capitalize()} Iteration {i}, Loss {loss.item()}, Scale {torch.exp(log_scale).item()}')

            loss.backward()
            
            if phase == 'scale_and_translation':
                print(f"log_scale grad: {log_scale.grad.item()}, t grad: {t.grad}")

            optimizer.step()

    # Final transformation matrix
    T = get_transformation_matrix(euler_angles, log_scale, t)
    return T.detach().numpy()

@click.command()
@click.option('--dataset', '-d', type=str, default='colmap', help='Dataset name (polycam, mvsnet, nerf, scannet, waymo)')
@click.argument('source_paths', nargs=-1, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output_dir', '-o', required=True, help='Path to the output directory')
@click.option('--init', default='colmap', type=str, help='Initializer name (colmap, loftr, dust3r, mvsplat, midas)')
@click.option('--overwrite', help='Overwrite existing files', is_flag=True)
@click.option('--resolution', '-r', default=1, type=int, help='Resolution')
@click.option('--num_images', '-n', default=5, type=int, help='Number of images to process')
def main(dataset, source_paths, output_dir, init, overwrite, resolution, num_images):
    # Split the comma-separated list of source paths into a list
    source_path_list = source_paths

    # Process each source path
    ref_dataset = datasets.make({"name": "colmap", "source_path": source_path_list[0]})
    ref_cameras = ref_dataset.all_cameras[:num_images]
    ref_extrinsics = np.concatenate([_camera.extrinsics[None] for _camera in ref_cameras])
    dust3r_initializer = initializers.make("dust3r")
    colmap_initializer = initializers.make({"name": "colmap", "workspace_dir": output_dir})
    
    dummy_gaussians = models.make({"name": "vanilla_pcd", "sh_degree": 1})
    
    src_datasets = []
    for source_path in tqdm(source_paths[:1], desc="Processing datasets"):
        src_dataset = datasets.make({"name": dataset, "source_path": source_path})
        src_cameras = src_dataset.all_cameras[:num_images]
        src_extrinsics = np.concatenate([_camera.extrinsics[None] for _camera in src_cameras])
        
        # Apply the initializer to the combined cameras
        dust3r_initializer(dummy_gaussians, ref_cameras + src_cameras)
        
        # ref_dataset.all_cameras = ref_cameras + src_cameras
        # colmap_initializer(dummy_gaussians, ref_dataset, overwrite=overwrite)
        # exit()
        
        # Get new extrinsics from the initializer
        common_ref_extrinsics = np.concatenate([_camera.extrinsics[None] for _camera in dust3r_initializer.cameras[:num_images]])
        common_src_extrinsics = np.concatenate([_camera.extrinsics[None] for _camera in dust3r_initializer.cameras[num_images:]])

        ref_transform = optimize_pose(ref_extrinsics, common_ref_extrinsics,)
        # src_transform = optimize_pose(src_extrinsics, common_src_extrinsics)
                
        common_ref_extrinsics = torch.bmm(torch.from_numpy(ref_extrinsics), 
                                          torch.from_numpy(ref_transform).unsqueeze(0).repeat(num_images, 1, 1)).cpu().numpy()
        # common_src_extrinsics = torch.bmm(torch.from_numpy(src_extrinsics), 
                                        #   torch.from_numpy(src_transform).unsqueeze(0).repeat(num_images, 1, 1)).cpu().numpy()

        for _ref_camera, _final_extrinsic in zip(ref_cameras, common_ref_extrinsics):
            _ref_camera.extrinsics = _final_extrinsic
                        
        for _src_camera, _final_extrinsic in zip(src_cameras, common_src_extrinsics):
            _src_camera.extrinsics = _final_extrinsic

        # Extend the ref_dataset cameras
        ref_dataset.all_cameras = ref_cameras + src_cameras

        # Initialize the combined dataset using colmap_initializer
        colmap_initializer(dummy_gaussians, ref_dataset, overwrite=overwrite)


if __name__ == '__main__':
    main()