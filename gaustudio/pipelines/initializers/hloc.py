import os
import numpy as np
import shutil
from pathlib import Path

from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.colmap import ColmapInitializer
from gaustudio.utils.colmap_utils import create_images_from_pose_dict

try:
    from hloc import extract_features, match_features, match_dense
    from hloc import triangulation, pairs_from_poses
    hloc_installed = True
except:
    hloc_installed = False


@initializers.register('hloc')
class HlocInitializer(ColmapInitializer):
    def process_dataset(self):
        if not hloc_installed:
            raise ImportError("Please install hloc to use HlocInitializer.")

        create_images_from_pose_dict(self.ws_dir, self.pose_dict)
        sfm_pairs = Path(f'{self.ws_dir}/pairs-sfm.txt')
        pairs_from_poses.main(Path(f'{self.ws_dir}/model'), sfm_pairs, num_matched=10)
        
        feature_conf = extract_features.confs["superpoint_aachen"]
        matcher_conf = match_features.confs["NN-superpoint"]
        features = extract_features.main(
            feature_conf, Path(self.images_dir) ,
            feature_path=Path(self.ws_dir) / "model" / 'features.h5' , as_half=False)
        sfm_matches = match_features.main(
            matcher_conf, sfm_pairs, features=Path(self.ws_dir) / "model" / 'features.h5',
            matches=Path(self.ws_dir) / "model" / 'matches.h5')

        sparse_reconstruction_folder = Path(self.ws_dir) / 'sparse' / '0'
        os.makedirs(sparse_reconstruction_folder, exist_ok=True)
        triangulation.main(sparse_reconstruction_folder, Path(f'{self.ws_dir}/model'), Path(self.images_dir),
                           sfm_pairs, features, sfm_matches, skip_geometric_verification=True)
        shutil.rmtree(os.path.join(self.ws_dir, 'model'))

@initializers.register('loftr')
class LoftrInitializer(ColmapInitializer):
    def process_dataset(self):
        if not hloc_installed:
            raise ImportError("Please install hloc to use LoftrInitializer.")

        create_images_from_pose_dict(self.ws_dir, self.pose_dict)
        sfm_pairs = Path(f'{self.ws_dir}/pairs-sfm.txt')
        pairs_from_poses.main(Path(f'{self.ws_dir}/model'), sfm_pairs, num_matched=10)
        matcher_conf = match_dense.confs['loftr']
        features, sfm_matches = match_dense.main(matcher_conf, sfm_pairs, Path(self.images_dir),
                                         self.ws_dir, max_kps=8192)

        sparse_reconstruction_folder = Path(self.ws_dir) / 'sparse' / '0'
        os.makedirs(sparse_reconstruction_folder, exist_ok=True)
        triangulation.main(sparse_reconstruction_folder, Path(f'{self.ws_dir}/model'), Path(self.images_dir), sfm_pairs, features, sfm_matches)


@initializers.register('loftr')
class LoftrInitializer(ColmapInitializer):
    def process_dataset(self):
        if not hloc_installed:
            raise ImportError("Please install hloc to use LoftrInitializer.")

        create_images_from_pose_dict(self.ws_dir, self.pose_dict)
        sfm_pairs = Path(f'{self.ws_dir}/pairs-sfm.txt')
        pairs_from_poses.main(Path(f'{self.ws_dir}/model'), sfm_pairs, num_matched=10)
        matcher_conf = match_dense.confs['loftr']
        features, sfm_matches = match_dense.main(matcher_conf, sfm_pairs, Path(self.images_dir),
                                         self.ws_dir, max_kps=8192)

        sparse_reconstruction_folder = Path(self.ws_dir) / 'sparse' / '0'
        os.makedirs(sparse_reconstruction_folder, exist_ok=True)
        triangulation.main(sparse_reconstruction_folder, Path(f'{self.ws_dir}/model'), Path(self.images_dir), sfm_pairs, features, sfm_matches)