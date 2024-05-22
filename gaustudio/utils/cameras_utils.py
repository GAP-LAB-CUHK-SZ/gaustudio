import numpy as np
from gaustudio.datasets import Camera
import math

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def JSON_to_camera(camera_json, data_device=None):
    id = camera_json['id']
    image_name = camera_json['img_name']
    width = camera_json['width']
    height = camera_json['height']
    position = np.array(camera_json['position'])
    rotation = np.array(camera_json['rotation'])
    
    W2C = np.eye(4)
    W2C[:3, :3] = rotation
    W2C[:3, 3] = position
    Rt = np.linalg.inv(W2C)
    
    rotation = Rt[:3, :3]
    position = Rt[:3, 3]
    
    fy = camera_json['fy']
    fx = camera_json['fx']
    
    R = rotation.transpose()
    T = position
    
    return Camera(
        image_name=image_name,
        image_width=width,
        image_height=height,
        R=R,
        T=T,
        FoVx=focal2fov(fx, width),
        FoVy=focal2fov(fy, height)
    )
    
