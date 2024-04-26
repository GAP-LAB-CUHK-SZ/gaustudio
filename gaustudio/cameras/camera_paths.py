import json
import copy 
import torch
from gaustudio.utils.cameras_utils import JSON_to_camera
from gaustudio.utils.pose_utils import get_interpolated_poses, quaternion_from_matrix, quaternion_matrix

def get_path_from_json(json_path):
    print("Loading camera data from {}".format(json_path))
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    cameras = []
    for camera_json in camera_data:
        camera = JSON_to_camera(camera_json, "cuda")
        cameras.append(camera)
    return cameras

def upsample_cameras_velocity(cameras, meters_per_frame=0.1, angles_per_frame=1):
    """
    Upsample cameras to achieve a target average velocity.

    Args:
        cameras (list): List of camera objects with extrinsics (transformation matrices).
        meters_per_frame (float): Target average velocity (meters per frame).
        angles_per_frame (float): Target average angular velocity (degrees per frame).

    Returns:
        list: List of upsampled camera objects.
    """
    new_cameras = []
    total_idx = 0
    for idx in range(len(cameras) - 1):
        current_camera = cameras[idx]
        next_camera = cameras[idx + 1]

        translation_change = np.linalg.norm(next_camera.extrinsics[:3, 3] - current_camera.extrinsics[:3, 3])
        rotation_change = np.rad2deg(np.arccos(np.clip((np.trace(np.dot(current_camera.extrinsics[:3, :3].T, next_camera.extrinsics[:3, :3])) - 1) / 2, -1.0, 1.0)))
        steps_per_transition_translation = max(int(translation_change / meters_per_frame), 1)
        steps_per_transition_rotation = max(int(rotation_change / angles_per_frame), 1)
        steps_per_transition = max(steps_per_transition_translation, steps_per_transition_rotation)

        intermediate_cameras = get_interpolated_poses(current_camera.extrinsics, next_camera.extrinsics, steps=steps_per_transition)

        for intermediate_camera in intermediate_cameras:
            view_new = copy.deepcopy(current_camera)
            view_new.extrinsics = intermediate_camera
            view_new.image_name = str(total_idx).zfill(8)
            new_cameras.append(view_new)
            total_idx += 1

    return new_cameras
def downsample_cameras(cameras, translation_threshold=0.1, rotation_threshold=15, min_samples=10):
    """
    Downsample cameras based on translation and rotation thresholds.
    Ensure a minimum number of samples are retained.

    Args:
        cameras (list): List of camera objects with extrinsics (translation and rotation).
        translation_threshold (float): Maximum translation change (in meters) between keyframes.
        rotation_threshold (float): Maximum rotation change (in degrees) between keyframes.
        min_samples (int): Minimum number of samples to retain after downsampling.

    Returns:
        list: List of downsampled camera objects.
    """
    downsampled_cameras = []
    prev_camera = None

    # Convert rotation_threshold from degrees to radians
    rotation_threshold = np.deg2rad(rotation_threshold)

    # Ensure at least min_samples are retained
    if len(cameras) <= min_samples:
        return cameras

    for camera in cameras:
        if prev_camera is None:
            downsampled_cameras.append(camera)
            prev_camera = camera
            continue

        # Calculate translation change
        translation_change = np.linalg.norm(camera.extrinsics[:3, 3] - prev_camera.extrinsics[:3, 3])

        # Calculate rotation change
        prev_rotmat = prev_camera.extrinsics[:3, :3]
        curr_rotmat = camera.extrinsics[:3, :3]
        rotation_change = np.arccos((np.trace(prev_rotmat.T @ curr_rotmat) - 1) / 2)

        # Check if translation or rotation change exceeds the threshold
        if translation_change > translation_threshold or rotation_change > rotation_threshold:
            downsampled_cameras.append(camera)
            prev_camera = camera

    # If the number of downsampled cameras is less than min_samples, return the original cameras
    if len(downsampled_cameras) < min_samples:
        return cameras

    return downsampled_cameras

import numpy as np

def validate_paths(cameras, window_size_ratio=0.1, speed_tolerance=0.2, discard_outliers=True):
    """
    Validate camera paths based on average speed within a sliding window.
    Optionally discard outlier cameras.

    Args:
        cameras (list): List of camera objects with extrinsics (translation and rotation).
        window_size_ratio (float): Ratio of the window size to the total number of cameras.
        speed_tolerance (float): Tolerance factor for determining the average speed threshold.
        discard_outliers (bool): Whether to discard outlier cameras or keep them.

    Returns:
        tuple: (valid_cameras, invalid_cameras)
            valid_cameras (list): List of camera objects with valid average speed.
            invalid_cameras (list): List of camera objects with invalid average speed.
    """
    valid_cameras = []
    invalid_cameras = []
    prev_camera = None

    num_cameras = len(cameras)
    window_size = max(3, int(num_cameras * window_size_ratio))  # Minimum window size of 3

    for i, camera in enumerate(cameras):
        if prev_camera is None:
            valid_cameras.append(camera)
            prev_camera = camera
            continue

        # Calculate translation change
        translation_change = np.linalg.norm(camera.extrinsics[:3, 3] - prev_camera.extrinsics[:3, 3])

        # Calculate average speed
        avg_speed = translation_change / (i - (i - 1))  # Assuming a constant time difference of 1

        # Calculate average speed threshold based on a sliding window
        window_start = max(0, i - window_size)
        window_end = i + 1
        window_cameras = cameras[window_start:window_end]
        window_speeds = [np.linalg.norm(cam.extrinsics[:3, 3] - cameras[max(0, j - 1)].extrinsics[:3, 3]) /
                         (j - max(0, j - 1))  # Assuming a constant time difference of 1
                         for j, cam in enumerate(window_cameras, start=window_start)]
        avg_speed_threshold = np.mean(window_speeds) * (1 + speed_tolerance)

        # Check if average speed exceeds the threshold
        if avg_speed > avg_speed_threshold:
            if discard_outliers:
                continue  # Discard the camera if it's an outlier
            else:
                invalid_cameras.append(camera)
        else:
            valid_cameras.append(camera)

        prev_camera = camera

    return valid_cameras, invalid_cameras

from scipy.signal import savgol_filter
import numpy as np
import numpy.linalg as la
def unwrap_quaternions(qvecs):
    qvecs_unwrapped = np.zeros_like(qvecs)
    qvecs_unwrapped[0] = qvecs[0]
    for i in range(1, qvecs.shape[0]):
        dot = np.clip(np.sum(qvecs_unwrapped[i-1] * qvecs[i]), -1.0, 1.0)
        qvecs_unwrapped[i] = (qvecs[i] if dot > 0 else -qvecs[i])
    return qvecs_unwrapped

def smoothen_cameras(cameras, window_size_ratio=0.1, polyorder=2):
    num_cameras = len(cameras)
    window_size = max(3, int(num_cameras * window_size_ratio))  # Minimum window size of 3

    new_cameras = []
    total_idx = 0
    translates = torch.stack([camera.extrinsics[:3, 3] for camera in cameras]).cpu().numpy()
    qvecs = np.stack([quaternion_from_matrix(camera.extrinsics[:3, :3].cpu().numpy()) for camera in cameras])
    qvecs = unwrap_quaternions(qvecs)
    for dim in range(translates.shape[1]):
        translates[:, dim] = savgol_filter(translates[:, dim], window_size, polyorder)
    
    for dim in range(qvecs.shape[1]):
        qvecs[:, dim] = savgol_filter(qvecs[:, dim], window_size, polyorder)

    for camera, smooth_translate, smooth_qvec in zip(cameras, translates, qvecs):
        smooth_qvec /= la.norm(smooth_qvec)  # Normalize quaternion
        camera_new = copy.deepcopy(camera)
        updated_extrinsics = quaternion_matrix(smooth_qvec)
        updated_extrinsics[:3, 3] = smooth_translate
        camera_new.extrinsics = updated_extrinsics
        new_cameras.append(camera_new)
        total_idx += 1

    return new_cameras