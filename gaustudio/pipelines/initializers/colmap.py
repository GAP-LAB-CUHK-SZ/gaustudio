import os
import pycolmap
import numpy as np
import torchvision
from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.base import BaseInitializer
from gaustudio.utils.colmap_utils import COLMAPDatabase, create_images_bin, create_cameras_and_points_bin

@initializers.register('colmap')
class ColmapInitializer(BaseInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        self.db_path = os.path.join(self.ws_dir, "database.db")
        self.images_dir = f'{self.ws_dir}/images'
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.pose_dict = {}
    
    def __call__(self, pcd, dataset, overwrite=False):
        # Load ply file if available
        if dataset.ply_path is not None and os.path.exists(dataset.ply_path) and not overwrite: 
            pcd.load(dataset.ply_path)

        # Skip processing if sparse results exists
        if not os.path.exists(f'{self.ws_dir}/sparse') or overwrite:
            self.preprocess(dataset)
            self.process()

        pcd = self.postprocess(pcd)
        return pcd

    def preprocess(self, dataset):
        print("Caching images...")
        for img_id, camera in enumerate(dataset):
            img_name = str(img_id).zfill(8)
            torchvision.utils.save_image(camera.image.permute(2, 0, 1), f'{self.ws_dir}/images/{img_name}.jpg')
            self.pose_dict[img_name] = camera.extrinsics.inverse()
            intrinsics_dict = {'width': camera.image_width, 'height': camera.image_height, 
                      'fx': camera.intrinsics[0, 0], 'fy': camera.intrinsics[1, 1],
                      'cx': camera.intrinsics[0, 2], 'cy': camera.intrinsics[1, 2]}
        
        # TODO: Support several camera model        
        print("Creating cameras and points model...")
        create_cameras_and_points_bin(self.ws_dir, intrinsics_dict)

    def process(self):
        os.remove(self.db_path) if os.path.exists(self.db_path) else None
        pycolmap.extract_features(
            image_path=self.images_dir,
            database_path=self.db_path
        )
        pycolmap.match_sequential(self.db_path)

        db = COLMAPDatabase.connect(f'{self.ws_dir}/database.db')
        images = list(db.execute('select * from images'))
        create_images_bin(self.ws_dir, self.pose_dict, images)
        
        sparse_reconstruction_folder = os.path.join(self.ws_dir, 'sparse', '0')
        os.makedirs(sparse_reconstruction_folder, exist_ok=True)

        reference = pycolmap.Reconstruction(f'{self.ws_dir}/model')
        pycolmap.triangulate_points(reference, f'{self.ws_dir}/database.db', self.images_dir, sparse_reconstruction_folder)

    def postprocess(self, pcd):
        from gaustudio.utils.colmap_utils import read_points3D_binary
        candidates = ['sparse/0/points3D.bin', 'sparse/points3D.bin']
        for candidate in candidates:
            candidate_path = os.path.join(self.ws_dir, candidate)
            if os.path.exists(candidate_path):
                pts3d = read_points3D_binary(candidate_path)
                xyz = np.array([pts3d[k].xyz for k in pts3d])
                normal = np.zeros_like(xyz)
                rgb = np.array([pts3d[k].rgb for k in pts3d])
                pcd.update(xyz=xyz, normal=normal, rgb=rgb)
                break
        return pcd
