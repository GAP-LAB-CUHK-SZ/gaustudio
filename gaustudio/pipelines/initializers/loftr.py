import os
import pycolmap
import numpy as np
import torchvision
from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.colmap import ColmapInitializer

from gaustudio.utils.colmap_utils import create_images_from_pose_dict

@initializers.register('loftr')
class LoftrInitializer(ColmapInitializer):
    def process(self):
        from pathlib import Path
        from hloc import match_dense, triangulation, pairs_from_poses
        
        create_images_from_pose_dict(self.ws_dir, self.pose_dict)
        sfm_pairs = Path(f'{self.ws_dir}/pairs-sfm.txt')
        pairs_from_poses.main(Path(f'{self.ws_dir}/model'), sfm_pairs, num_matched=10)
        matcher_conf = match_dense.confs['loftr']
        features, sfm_matches = match_dense.main(matcher_conf, sfm_pairs, Path(self.images_dir),
                                         self.ws_dir, max_kps=8192)

        sparse_reconstruction_folder = Path(self.ws_dir) / 'sparse' / '0'
        os.makedirs(sparse_reconstruction_folder, exist_ok=True)
        triangulation.main(sparse_reconstruction_folder, Path(f'{self.ws_dir}/model'), Path(self.images_dir), sfm_pairs, features, sfm_matches)
