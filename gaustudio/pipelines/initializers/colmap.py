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
        self.images_dir = os.path.join(self.ws_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.pose_dict = {}
    
    def __call__(self, model, dataset, overwrite=False):
        # Load ply file if available
        if dataset.ply_path is not None and os.path.exists(dataset.ply_path) and not overwrite: 
            model.load(dataset.ply_path)
            return model

        # Skip processing if sparse results exists
        if not os.path.exists(f'{self.ws_dir}/sparse') or overwrite:
            self.cache_dataset(dataset)
            self.process_dataset()

        model = self.build_model(model)
        return model

    def cache_dataset(self, dataset):
        print("Caching images and preparing dataset...")
        for img_id, camera in enumerate(dataset):
            img_name = str(img_id).zfill(8)
            torchvision.utils.save_image(camera.image.permute(2, 0, 1), os.path.join(self.images_dir, f'{img_name}.jpg'))
            
            if camera.mask is not None:
                self.masks_dir = os.path.join(self.ws_dir, 'masks')
                os.makedirs(self.masks_dir, exist_ok=True)
                torchvision.utils.save_image(camera.mask.float(), os.path.join(self.masks_dir, f'{img_name}.png'))

            self.pose_dict[img_name] = camera.extrinsics.inverse()
            intrinsics = {
                'width': camera.image_width, 'height': camera.image_height, 
                'fx': camera.intrinsics[0, 0], 'fy': camera.intrinsics[1, 1],
                'cx': camera.intrinsics[0, 2], 'cy': camera.intrinsics[1, 2]
            }

        print("Creating camera and points model data...")
        create_cameras_and_points_bin(self.ws_dir, intrinsics)

    def process_dataset(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        pycolmap.extract_features(image_path=self.images_dir, database_path=self.db_path)
        pycolmap.match_exhaustive(self.db_path)
        
        db = COLMAPDatabase.connect(self.db_path)
        images = list(db.execute('select * from images'))
        create_images_bin(self.ws_dir, self.pose_dict, images)

        sparse_reconstruction_folder = os.path.join(self.ws_dir, 'sparse', '0')
        os.makedirs(sparse_reconstruction_folder, exist_ok=True)

        reference = pycolmap.Reconstruction(os.path.join(self.ws_dir, 'model'))
        pycolmap.triangulate_points(reference, self.db_path, self.images_dir, sparse_reconstruction_folder)

    def build_model(self, model):
        """
        Build the 3D model point cloud by reading the COLMAP points3D binary file.
        :param model: model instance to update
        :return: Updated model instance
        """
        candidates = ['sparse/0/points3D.bin', 'sparse/points3D.bin']
        from gaustudio.utils.colmap_utils import read_points3D_binary
        
        for candidate in candidates:
            candidate_path = os.path.join(self.ws_dir, candidate)
            if os.path.exists(candidate_path):
                try:
                    pts3d = read_points3D_binary(candidate_path)
                except Exception as e:
                    print(f"Failed to read points3D binary file: {e}")
                    raise

                try:
                    xyz = np.array([pts3d[k].xyz for k in pts3d])
                    rgb = np.array([pts3d[k].rgb / 255 for k in pts3d])
                    model.create_from_attribute(xyz=xyz, rgb=rgb)
                except Exception as e:
                    print(f"Failed to update point cloud: {e}")
                    raise
                break
        else:
            print(f"No points3D binary file found in {self.ws_dir} with candidates {candidates}")
            raise
        return model
