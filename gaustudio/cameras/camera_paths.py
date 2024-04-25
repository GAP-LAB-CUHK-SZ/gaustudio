import json
import copy 
from gaustudio.utils.cameras_utils import JSON_to_camera
from gaustudio.utils.pose_utils import get_interpolated_poses

def get_path_from_json(json_path):
    print("Loading camera data from {}".format(json_path))
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    cameras = []
    for camera_json in camera_data:
        camera = JSON_to_camera(camera_json, "cuda")
        cameras.append(camera)
    return cameras

def get_interpolated_cameras(cameras, steps_per_transition=30):
    new_cameras = []
    total_idx = 0
    for idx in range(len(cameras)-1):
        current_camera = cameras[idx]
        next_camera = cameras[idx+1]
        
        intermediate_cameras = get_interpolated_poses(current_camera.extrinsics.cpu().numpy(), 
                                          next_camera.extrinsics.cpu().numpy(), steps=steps_per_transition)
        for intermediate_camera in intermediate_cameras:
            view_new = copy.deepcopy(current_camera)
            view_new.extrinsics = intermediate_camera
            view_new.image_name = str(total_idx).zfill(8)
            new_cameras.append(view_new)
            total_idx+=1
    return new_cameras
        