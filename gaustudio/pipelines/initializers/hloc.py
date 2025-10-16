import os
import numpy as np
import shutil
from pathlib import Path

from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.colmap import ColmapInitializer
from gaustudio.utils.colmap_utils import create_images_from_pose_dict

try:
    from hloc import extract_features, match_features, match_dense
    from hloc import triangulation
    hloc_installed = True
except:
    hloc_installed = False

def get_pairwise_distances(T_q2w, T_r2w):
    R_q2w = np.stack([T.r.as_matrix() for T in T_q2w]).astype(np.float32)
    t_q2w = np.stack([T.t for T in T_q2w]).astype(np.float32)
    R_r2w = np.stack([T.r.as_matrix() for T in T_r2w]).astype(np.float32)
    t_r2w = np.stack([T.t for T in T_r2w]).astype(np.float32)

    # equivalent to scipy.spatial.distance.cdist but supports fp32
    dt = t_q2w.dot(t_r2w.T)
    dt *= -2
    dt += np.einsum('ij,ij->i', t_q2w, t_q2w)[:, None]
    dt += np.einsum('ij,ij->i', t_r2w, t_r2w)[None]
    np.clip(dt, a_min=0, a_max=None, out=dt)  # numerical errors
    np.sqrt(dt, out=dt)

    trace = np.einsum('nji,mji->nm', R_q2w, R_r2w, optimize=True)
    dR = np.clip((trace - 1) / 2, -1., 1.)
    dR = np.rad2deg(np.abs(np.arccos(dR)))
    return dR, dt


def pairs_from_poses(
    images,
    overlap: int = 5,
    loop_Rt_thresh=(30.0, 2.0),      # (max rotation deg, max translation) for loop closure
    near_Rt_min_thresh=(1.0, 0.05),  # (min rotation deg, min translation) discard too-near
    max_loops_per_image: int = 5,
):
    """
    Build image pairs from a COLMAP images dict (id -> Image):
      1. Sequential (window = overlap) in the chosen ordering (by image id)
      2. Loop-closure candidates via pose proximity
      3. Discard pairs that are too near in BOTH rotation & translation
    """
    # Deterministic ordering by image id
    ordered = sorted(images.items(), key=lambda x: x[0])
    names = [im.name for _, im in ordered]

    # Collect world-to-camera poses (R_w2c, t_w2c)
    R_w2c = []
    t_w2c = []
    for _, im in ordered:
        R_w2c.append(im.qvec2rotmat())
        t_w2c.append(im.tvec)
    R_w2c = np.stack(R_w2c, 0).astype(np.float32)
    t_w2c = np.stack(t_w2c, 0).astype(np.float32)

    # Invert to camera-to-world: R_c2w = R_w2c^T, t_c2w = -R_c2w * t_w2c
    R_c2w = R_w2c.transpose(0, 2, 1)
    t_c2w = -(R_c2w @ t_w2c[:, :, None])[:, :, 0]

    N = len(names)
    if N == 0:
        return []

    R_loop_max, t_loop_max = loop_Rt_thresh
    R_near_min, t_near_min = near_Rt_min_thresh

    # Pairwise translation distances (cdist but fp32 + in-place)
    dt = t_c2w @ t_c2w.T
    dt *= -2
    sq = np.einsum('ij,ij->i', t_c2w, t_c2w)
    dt += sq[:, None]
    dt += sq[None]
    np.clip(dt, 0, None, out=dt)
    np.sqrt(dt, out=dt)  # (N,N)

    # Pairwise rotation (angle in deg)
    trace = np.einsum('nji,mji->nm', R_c2w, R_c2w, optimize=True)
    dR = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    dR = np.rad2deg(np.abs(np.arccos(dR)))  # (N,N)

    pairs = []
    added = set()

    # 1. Sequential window
    for i in range(N - 1):
        for j in range(i + 1, min(i + overlap + 1, N)):
            if dR[i, j] < R_near_min and dt[i, j] < t_near_min:
                continue
            key = (i, j)
            if key not in added:
                pairs.append((names[i], names[j]))
                added.add(key)

    # 2. Loop closures (outside sequential window)
    for i in range(N):
        start = i + overlap + 1
        if start >= N:
            continue
        cand_idx = np.arange(start, N)

        valid = (dR[i, cand_idx] < R_loop_max) & (dt[i, cand_idx] < t_loop_max)
        not_too_near = ~((dR[i, cand_idx] < R_near_min) & (dt[i, cand_idx] < t_near_min))
        valid &= not_too_near
        if not np.any(valid):
            continue

        vc = cand_idx[valid]
        order = np.lexsort((dR[i, vc], dt[i, vc]))  # prioritize closer translation then rotation
        vc = vc[order][:max_loops_per_image]

        for j in vc:
            key = (i, j)
            if key not in added:
                pairs.append((names[i], names[j]))
                added.add(key)

    return pairs

from hloc.utils.read_write_model import read_images_binary
def pairs_from_poses_main(model, output, overlap=5):
    images = read_images_binary(model / 'images.bin')
    
    pairs = pairs_from_poses(images, overlap=overlap)
    print(images)
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

@initializers.register('hloc')
class HlocInitializer(ColmapInitializer):
    def process_dataset(self):
        if not hloc_installed:
            raise ImportError("Please install hloc to use HlocInitializer.")

        create_images_from_pose_dict(self.ws_dir, self.pose_dict, self.filename_lookup)
        sfm_pairs = Path(f'{self.ws_dir}/pairs-sfm.txt')
        
        
        feature_conf = extract_features.confs["superpoint_aachen"]
        pairs_from_poses_main(Path(f'{self.ws_dir}/model'), sfm_pairs, overlap=10)
        matcher_conf = match_features.confs["superpoint+lightglue"]
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

        create_images_from_pose_dict(self.ws_dir, self.pose_dict, self.filename_lookup)
        sfm_pairs = Path(f'{self.ws_dir}/pairs-sfm.txt')
        pairs_from_poses.main(Path(f'{self.ws_dir}/model'), sfm_pairs, num_matched=10)
        matcher_conf = match_dense.confs['loftr']
        features, sfm_matches = match_dense.main(matcher_conf, sfm_pairs, Path(self.images_dir),
                                         self.ws_dir, max_kps=8192)

        sparse_reconstruction_folder = Path(self.ws_dir) / 'sparse' / '0'
        os.makedirs(sparse_reconstruction_folder, exist_ok=True)
        triangulation.main(sparse_reconstruction_folder, Path(f'{self.ws_dir}/model'), Path(self.images_dir), sfm_pairs, features, sfm_matches)
