import os
import cv2
import numpy as np
from gaustudio import datasets
from gaustudio.datasets.base import BaseDataset
from gaustudio.datasets.utils import focal2fov
from typing import Dict
from tqdm import tqdm

def load_cam(file: str, max_d=256, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    with open(file) as f:
        words = f.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (cam[1][3][2] - 1)
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (cam[1][3][2] - 1)
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam[0], cam[1]

class MvsnetDatasetBase(BaseDataset):
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.image_dir = self.source_path / "blended_images"
        if self.image_dir.exists() is False:
            self.image_dir = self.source_path / "images"
        if self.image_dir.exists() is False:
            raise ValueError("No image directory found")
        self.camera_dir = self.source_path /  "cams" 
        self.image_filenames = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)],
                                      key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0]))

        self._initialize()
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        for image_path in tqdm(self.image_filenames, total=len(self.image_filenames), desc="Reading cameras"):        
            image = cv2.imread(str(image_path))
            height, width, _ = image.shape
            
            _id = os.path.splitext(os.path.basename(image_path))[0]
            cam_file_path = self.camera_dir / ('%s_cam.txt' % _id)
            extrinsic, intr = load_cam(cam_file_path)
            fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]                
            
            R = np.transpose(extrinsic[:3, :3])
            T = extrinsic[:3, 3]
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, 
                                      image_path=image_path, image_width=width, image_height=height,
                                      principal_point_ndc=np.array([cx / width, cy /height]))
            all_cameras_unsorted.append(_camera)
        self.finalize_cameras(all_cameras_unsorted)
            
@datasets.register('mvsnet')
class MvsnetDataset(MvsnetDatasetBase):
    pass
