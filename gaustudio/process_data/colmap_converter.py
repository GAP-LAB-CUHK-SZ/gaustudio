import os
import pycolmap
import cv2
import numpy as np

from gaustudio.process_data.base import BaseConverter
from gaustudio.process_data.polycam_utils import PolycamDataset
from gaustudio.process_data.colmap_utils import rotmat2qvec, qvec2rotmat, COLMAPDatabase, read_points3D_binary, read_cameras_binary, read_images_binary


def compute_scale_and_shift_ls(prediction, target, mask):
    # tuple specifying with axes to sum
    sum_axes = (0, 1)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, sum_axes)
    a_01 = np.sum(mask * prediction, sum_axes)
    a_11 = np.sum(mask, sum_axes)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, sum_axes)
    b_1 = np.sum(mask * target, sum_axes)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def apply_depth_colormap(depth: np.ndarray, near_plane=None, far_plane=None): 
    near_plane = near_plane or np.min(depth)
    far_plane = far_plane or np.max(depth)
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = np.clip(depth, 0, 1)
    import matplotlib.pyplot as plt
    colored_image = plt.cm.viridis(depth)[:, :, :3]

    return colored_image

import torch
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to("cuda").eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
def predict_depth_single(input_image, input_sparse_depth, device="cuda"):
    input_height, input_width = np.shape(input_image)[0], np.shape(input_image)[1]
    input_sparse_depth_valid = (input_sparse_depth > 0).astype(bool)

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_batch = midas_transforms(input_image).to(device)

    with torch.no_grad():
        depth_pred = midas(input_batch)

        depth_pred = (
                torch.nn.functional.interpolate(
                    depth_pred.unsqueeze(1),
                    size=(input_height, input_width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
    return depth_pred

class LeastSquaresEstimator(object):
    def __init__(self, estimate, target, valid):
        self.estimate = estimate
        self.target = target
        self.valid = valid

        # to be computed
        self.scale = 1.0
        self.shift = 0.0
        self.output = None

    def compute_scale_and_shift(self):
        self.scale, self.shift = compute_scale_and_shift_ls(self.estimate, self.target, self.valid)

    def apply_scale_and_shift(self):
        self.output = self.estimate * self.scale + self.shift

    def clamp_min_max(self, clamp_min=None, clamp_max=None):
        if clamp_min is not None:
            if clamp_min > 0:
                clamp_min_inv = 1.0/clamp_min
                self.output[self.output > clamp_min_inv] = clamp_min_inv
                assert np.max(self.output) <= clamp_min_inv
            else: # divide by zero, so skip
                pass
        if clamp_max is not None:
            clamp_max_inv = 1.0/clamp_max
            self.output[self.output < clamp_max_inv] = clamp_max_inv
            # print(np.min(self.output), clamp_max_inv)
            assert np.min(self.output) >= clamp_max_inv

def create_cameras_and_points_txt(target: str, intrinsic_dict):
    os.makedirs(f'{target}/model', exist_ok=True)
    os.system(f'touch {target}/model/points3D.txt')
    with open(f'{target}/model/cameras.txt', 'w') as f:
        f.write(f'1 PINHOLE {intrinsic_dict["width"]} {intrinsic_dict["height"]} {intrinsic_dict["fx"]} {intrinsic_dict["fy"]} {intrinsic_dict["cx"]} {intrinsic_dict["cy"]}')


def create_images_txt(target: str, pose_dict: dict, images: list):
    data_list = []
    for image in images:
        img_name = image[1][:-4]
        print(img_name)
        rt = pose_dict[img_name]
        rt = np.linalg.inv(rt)
        r = rt[:3, :3]
        t = rt[:3, 3]
        q = rotmat2qvec(r)
        data = [image[0], *q, *t, 1, f'{img_name}.jpg']
        data = [str(_) for _ in data]
        data = ' '.join(data)
        data_list.append(data)

    with open(f'{target}/model/images.txt', 'w') as f:
        for data in data_list:
            f.write(data)
            f.write('\n\n')

class ColmapConverter(BaseConverter):
    def __init__(self, dataset, workspace_dir):
        super().__init__(dataset, workspace_dir)
        self.db_path = os.path.join(workspace_dir, "database.db")
        self.ws_dir = workspace_dir
        self.images_dir = f'{self.ws_dir}/images'
        self.depths_dir = f'{self.ws_dir}/depths'
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.depths_dir, exist_ok=True)
        
        self.pose_dict = {}
    def preprocess(self):
        print("Caching images...")
        for img_id, data in enumerate(self.dataset):
            img_name = str(img_id).zfill(4)
            cv2.imwrite(f'{self.ws_dir}/images/{img_name}.jpg', data['color'])
            cv2.imwrite(f'{self.ws_dir}/depths/{img_name}.png', data['depth'])
            self.pose_dict[img_name] = data['c2w']
        print("Creating camera model...")
        create_cameras_and_points_txt(self.ws_dir, data['intrinsic_dict'])

    def process(self):
        os.remove(self.db_path) if os.path.exists(self.db_path) else None
        pycolmap.extract_features(
            image_path=self.images_dir,
            database_path=self.db_path
        )
        pycolmap.match_exhaustive(self.db_path)

        db = COLMAPDatabase.connect(f'{self.ws_dir}/database.db')
        images = list(db.execute('select * from images'))
        create_images_txt(self.ws_dir, self.pose_dict, images)
        
        sparse_reconstruction_folder = os.path.join(self.ws_dir, 'sparse', '0')
        os.makedirs(sparse_reconstruction_folder, exist_ok=True)

        reference = pycolmap.Reconstruction(f'{self.ws_dir}/model')
        pycolmap.triangulate_points(reference, f'{self.ws_dir}/database.db', self.images_dir, sparse_reconstruction_folder)

    def postprocess(self, max_repoj_err:float = 2.5, min_n_visible: int = 3, depth_scale_to_integer_factor: int=1000):
        
        ptid_to_info = read_points3D_binary(os.path.join(self.ws_dir, 'sparse', '0',  "points3D.bin"))
        cam_id_to_camera = read_cameras_binary(os.path.join(self.ws_dir, 'sparse', '0',  "cameras.bin"))
        im_id_to_image = read_images_binary(os.path.join(self.ws_dir, 'sparse', '0', "images.bin"))

        CAMERA_ID = 1
        W = cam_id_to_camera[CAMERA_ID].width
        H = cam_id_to_camera[CAMERA_ID].height

        iter_images = iter(im_id_to_image.items())
        os.makedirs(os.path.join(self.ws_dir, "depths"), exist_ok=True) 
        for im_id, im_data in iter_images:
            pids = [pid for pid in im_data.point3D_ids if pid != -1]
            xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])         
            
            if len(xyz_world) < min_n_visible:
                dense_depth = np.zeros((H, W), dtype=np.uint16)
            else:
                rotation = qvec2rotmat(im_data.qvec)
                z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
                errors = np.array([ptid_to_info[pid].error for pid in pids])
                n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])
                uv = np.array([im_data.xys[i] for i in range(len(im_data.xys)) if im_data.point3D_ids[i] != -1])

                idx = np.where(
                    (errors <= max_repoj_err)
                    & (n_visible >= min_n_visible)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < W)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < H)
                )
                z = z[idx]
                uv = uv[idx]

                uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
                depth = np.zeros((H, W), dtype=np.float32)
                depth[vv, uu] = z
                depth_valid = depth > 0.
                
                input_image = cv2.imread(os.path.join(self.ws_dir, 'images', str(im_data.name)))
                depth_pred = predict_depth_single(input_image, depth)
                GlobalAlignment = LeastSquaresEstimator(
                    estimate=depth_pred,
                    target=depth,
                    valid=depth_valid
                )
                GlobalAlignment.compute_scale_and_shift()
                GlobalAlignment.apply_scale_and_shift()
                metric_depth = GlobalAlignment.output
                dense_depth = GlobalAlignment.output.astype(np.float32)
                
                dense_depth = (depth_scale_to_integer_factor * dense_depth).astype(np.uint16)
            out_name = str(im_data.name)
            depth_path = os.path.join(self.ws_dir, "depths" ,out_name)
            depth_path = depth_path.replace(".jpg", ".png")
            cv2.imwrite(str(depth_path), dense_depth)  # type: ignore