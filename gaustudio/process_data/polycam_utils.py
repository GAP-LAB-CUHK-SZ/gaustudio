from gaustudio.process_data.base import BaseDataset
import os
import json
import cv2
import numpy as np

class PolycamDataset(BaseDataset):
    def __init__(self, source, transform=None):
        self.image_dir = os.path.join(source, "keyframes", "images")
        self.depth_dir = os.path.join(source, "keyframes", "depth")
        self.cameras_dir = os.path.join(source, "keyframes", "cameras")
        
        self.image_filenames = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)],
                                      key=lambda fn: int(os.path.splitext(os.path.basename(fn))[0]))

        self.transform = transform
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]        
        ori_frame_id = int(os.path.splitext(os.path.basename(image_filename))[0])
        
        json_filename = os.path.join(self.cameras_dir, "{}.json".format(ori_frame_id))
        frame_json = json.load(open(json_filename))
        
        width, height = frame_json["width"], frame_json["height"]
        c2w = np.array([
            [frame_json["t_20"], frame_json["t_21"], frame_json["t_22"], frame_json["t_23"]],
            [frame_json["t_00"], frame_json["t_01"], frame_json["t_02"], frame_json["t_03"]],
            [frame_json["t_10"], frame_json["t_11"], frame_json["t_12"], frame_json["t_13"]],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        c2w[..., 2] *= -1
        c2w[..., 1] *= -1
        
        image = cv2.imread(image_filename)
        depth = cv2.imread(os.path.join(self.depth_dir, "{}.png".format(ori_frame_id)), -1)
        depth = cv2.resize(depth, (width, height))
        
        intrinsic_4x4 = np.array([[frame_json['fx'], 0, frame_json['cx']],
                              [0, frame_json['fy'], frame_json['cy']],
                              [0, 0, 1]])
        intrinsics = {'width': width, 'height': height, 'fx': frame_json['fx'], 'fy': frame_json['fy'], 'cx': frame_json['cx'], 'cy': frame_json['cy']}
        
        
        sample = {"color": image, 
                  "depth": depth,
                  "c2w": c2w,
                  "intrinsic_dict": intrinsics,
                  "intrinsic_4x4":intrinsic_4x4}
        
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
if __name__ == "__main__":
    dataset = PolycamDataset("/workspace/dataset/VerseeScan/11月21日下午6-46-poly")