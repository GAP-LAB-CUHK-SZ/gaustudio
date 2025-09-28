import cv2
import numpy as np
from gaustudio import datasets
from gaustudio.datasets.base import BaseDataset
from gaustudio.datasets.utils import focal2fov
from typing import Dict
import pickle

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


class NeRODatasetBase(BaseDataset):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.image_path = self.source_path
        
        self.image_ids = sorted([int(f.stem.split('-')[0]) for f in self.source_path.glob("*-camera.pkl")])


        self._initialize()
    
    def _initialize(self):
        all_cameras_unsorted = []
    
        for _id in self.image_ids:
            cam = read_pickle(self.source_path / f"{_id}-camera.pkl")
            
            image_path = self.image_path / f"{_id}.png"
            _image = cv2.imread(str(image_path))
            height, width, _ = _image.shape
            extrinsics = np.eye(4)
            
            extrinsics[:3, :] = cam[0].astype(np.float32)
            intr = cam[1].astype(np.float32)
            fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]                
            
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]
            
            FoVy = focal2fov(fy, height)
            FoVx = focal2fov(fx, width)
            _camera = datasets.Camera(R=R, T=T, FoVy=FoVy, FoVx=FoVx, image_path=image_path, 
                                      image_width=width, image_height=height,
                                      principal_point_ndc=np.array([cx / width, cy /height]))
            all_cameras_unsorted.append(_camera)
        self.finalize_cameras(all_cameras_unsorted)
    
@datasets.register('nero')
class NeRODataset(NeRODatasetBase):
    pass
