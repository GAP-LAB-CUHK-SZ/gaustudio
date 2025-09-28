import os
import json
import cv2
import numpy as np
from gaustudio import datasets
from gaustudio.datasets.base import BaseDataset
from gaustudio.datasets.utils import focal2fov
from typing import Dict

class NSVFDatasetBase(BaseDataset):
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.image_dir = self.source_path / "rgb"
        self.pose_dir = self.source_path /  "pose"
        self.intrinsic_path = self.source_path / 'intrinsics.txt'
        
        self.image_filenames = []
        for f in os.listdir(self.image_dir):
            if os.path.basename(f).startswith('0_train'):
                self.image_filenames.append(os.path.join(self.image_dir, f))
        self.image_filenames = sorted(self.image_filenames, key=lambda fn: os.path.splitext(os.path.basename(fn))[0].split('_')[-1])
        self._initialize()
    
    def _initialize(self):
        all_cameras_unsorted = []
        
        with open(self.intrinsic_path) as f:
            intrs = f.readline().split()
            fx = fy = float(intrs[0])
            cx = float(intrs[1])
            cy = float(intrs[2])
            
        for image_path in self.image_filenames:
            _id = os.path.splitext(os.path.basename(image_path))[0]
            _image = cv2.imread(image_path)
            height, width, _ = _image.shape

            c2w = np.loadtxt(self.pose_dir / f'{_id}.txt' )

            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image_path=image_path, 
                                      image_width=width, image_height=height,
                                      principal_point_ndc=np.array([cx / width, cy /height]))
            all_cameras_unsorted.append(_camera)
        self.finalize_cameras(all_cameras_unsorted)
    
@datasets.register('nsvf')
class NSVFDataset(NSVFDatasetBase):
    pass
