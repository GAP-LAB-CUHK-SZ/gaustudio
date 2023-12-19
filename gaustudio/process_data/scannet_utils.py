import os
import cv2 
import numpy as np
from base import BaseDataset

class ScannetDataset(BaseDataset):
    def __init__(self, source, transform=None):
        self.color_dir = os.path.join(source, 'color')
        self.depth_dir = os.path.join(source, 'depth')  
        self.pose_dir = os.path.join(source, 'pose')
        self.intrinsic_path = os.path.join(source, 'intrinsic', 'intrinsic_color.txt')
        
        self.id_list = sorted([int(fn[:-4]) for fn in os.listdir(self.color_dir) if fn.endswith('.jpg')])
        self.transform = transform
        
        intrinsic = np.loadtxt(self.intrinsic_path)  
        self.intrinsic_dict = {'width': intrinsic[0], 'height': intrinsic[1], 
                               'fx': intrinsic[2], 'fy': intrinsic[3],
                               'cx': intrinsic[4], 'cy': intrinsic[5]}
        
        self.cache = {}
        
    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, idx):
        id = self.id_list[idx]
        if id in self.cache:
            return self.cache[id]
            
        pose = np.loadtxt(os.path.join(self.pose_dir, '%d.txt' % id)) 
        color = cv2.imread(os.path.join(self.color_dir, '%d.jpg' % id))
        depth = cv2.imread(os.path.join(self.depth_dir, '%d.png' % id), -1)
          
        sample = {'pose': pose, 'color': color, 'depth': depth}
        
        if self.transform:
            sample = self.transform(sample)
            
        self.cache[id] = sample
        return sample