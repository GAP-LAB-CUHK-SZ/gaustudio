import os
import numpy as np
import shutil
from pathlib import Path

from gaustudio.pipelines import initializers
from gaustudio.pipelines.initializers.colmap import ColmapInitializer
from gaustudio.utils.colmap_utils import create_images_from_pose_dict, Camera

try:
    from hloc import extract_features, match_features, match_dense
    from hloc import triangulation
    from hloc import reconstruction
    from hloc.utils.read_write_model import (
        read_cameras_binary,
        read_cameras_text,
        read_images_binary,
        write_cameras_binary,
        write_cameras_text,
    )
    hloc_installed = True
except ImportError:
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

def pairs_from_poses_main(model, output, overlap=5):
    images = read_images_binary(model / 'images.bin')
    
    pairs = pairs_from_poses(images, overlap=overlap)
    print(images)
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


def _to_opencv_camera(camera):
    """Return camera converted to OPENCV model if supported."""
    if camera.model == "OPENCV":
        return camera, False

    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
        opencv_params = np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif camera.model == "SIMPLE_PINHOLE":
        focal = camera.params[0]
        fx = fy = focal
        cx = camera.params[1]
        cy = camera.params[2]
        opencv_params = np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif camera.model == "SIMPLE_RADIAL":
        focal = camera.params[0]
        fx = fy = focal
        cx = camera.params[1]
        cy = camera.params[2]
        k1 = camera.params[3] if len(camera.params) > 3 else 0.0
        opencv_params = np.array([fx, fy, cx, cy, k1, 0.0, 0.0, 0.0], dtype=np.float64)
    elif camera.model == "RADIAL":
        focal = camera.params[0]
        fx = fy = focal
        cx = camera.params[1]
        cy = camera.params[2]
        k1 = camera.params[3] if len(camera.params) > 3 else 0.0
        k2 = camera.params[4] if len(camera.params) > 4 else 0.0
        opencv_params = np.array([fx, fy, cx, cy, k1, k2, 0.0, 0.0], dtype=np.float64)
    else:
        # Unsupported model for automatic conversion; keep as-is.
        return camera, False

    return camera._replace(model="OPENCV", params=opencv_params), True


def convert_cameras_to_opencv(model_dir: Path) -> None:
    """Convert COLMAP cameras in ``model_dir`` to the OPENCV model if possible."""
    if 'read_cameras_binary' not in globals():
        raise ImportError("hloc read/write utilities are required to convert camera models")

    model_dir = Path(model_dir)
    changed = False

    camera_bin = model_dir / "cameras.bin"
    if camera_bin.exists():
        cameras = read_cameras_binary(camera_bin)
        updated = {}
        bin_changed = False
        for cam_id, camera in cameras.items():
            converted, did_change = _to_opencv_camera(camera)
            updated[cam_id] = converted
            bin_changed |= did_change
        if bin_changed:
            write_cameras_binary(updated, camera_bin)
            changed = True

    camera_txt = model_dir / "cameras.txt"
    if camera_txt.exists():
        cameras_txt = read_cameras_text(camera_txt)
        updated_txt = {}
        txt_changed = False
        for cam_id, camera in cameras_txt.items():
            converted, did_change = _to_opencv_camera(camera)
            updated_txt[cam_id] = converted
            txt_changed |= did_change
        if txt_changed:
            write_cameras_text(updated_txt, camera_txt)
            changed = True

    if changed:
        print(f"Converted COLMAP cameras in {model_dir} to OPENCV model")

@initializers.register('hloc')
class HlocInitializer(ColmapInitializer):
    def process_dataset(self):
        if not hloc_installed:
            raise ImportError("Please install hloc to use HlocInitializer.")

        create_images_from_pose_dict(self.ws_dir, self.pose_dict, self.filename_lookup)
        sfm_pairs = Path(f'{self.ws_dir}/pairs-sfm.txt')
        
        
        feature_conf = extract_features.confs["aliked-n16"]
        pairs_from_poses_main(Path(f'{self.ws_dir}/model'), sfm_pairs, overlap=10)
        matcher_conf = match_features.confs["aliked+lightglue"]
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


@initializers.register('hloc_opencv')
class HlocOpenCVInitializer(HlocInitializer):
    def __init__(self, initializer_config):
        super().__init__(initializer_config)
        self.num_calibration_frames = initializer_config.get('num_calibration_frames', 40)
        self.feature_batch_size = initializer_config.get('feature_batch_size', 64)
        self.use_half_precision = initializer_config.get('use_half_precision', True)

    def cache_dataset(self, dataset):
        if not hloc_installed:
            raise ImportError("Please install hloc to use HlocOpenCVInitializer.")
        super().cache_dataset(dataset)
        convert_cameras_to_opencv(Path(self.ws_dir) / 'model')

    def process_dataset(self):
        # Optimized bundle adjustment workflow with parallel execution
        print("="*60)
        print("RUNNING OPTIMIZED BUNDLE ADJUSTMENT WORKFLOW")
        print("="*60)
        import threading
        import traceback

        # Step 1: Sample frames for calibration
        sampled_names = self._sample_frames(self.num_calibration_frames)
        print(f"Step 1: Sampled {len(sampled_names)} frames for intrinsic calibration")

        # Step 2: Extract hloc features for ALL frames upfront (optimized)
        print("Step 2: Extracting hloc features for all frames...")
        feature_conf = extract_features.confs["aliked-n16"].copy()
        # Optimization: increase batch size for faster extraction
        if 'batch_size' not in feature_conf:
            feature_conf['batch_size'] = self.feature_batch_size
        features_h5 = Path(self.ws_dir) / "model" / 'features.h5'
        features = extract_features.main(
            feature_conf, Path(self.images_dir),
            feature_path=features_h5, as_half=self.use_half_precision  # Optimized: use FP16
        )
        print(f"  ✓ Features extracted to {features_h5}")

        # Step 3: Generate pairs for 60 sampled frames
        print("Step 3: Generating pairs for sampled frames...")
        create_images_from_pose_dict(self.ws_dir, self.pose_dict, self.filename_lookup)

        # Generate pairs only for sampled frames
        sampled_pairs = self._generate_pairs_for_sampled(sampled_names)
        sampled_pairs_file = Path(self.ws_dir) / 'pairs-sampled.txt'
        with open(sampled_pairs_file, 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in sampled_pairs))
        print(f"  ✓ Generated {len(sampled_pairs)} pairs for sampled frames")

        # Step 4: Match 60 sampled frames with hloc
        print("Step 4: Matching sampled frames with hloc...")
        matcher_conf = match_features.confs["aliked+lightglue"]
        matches_h5 = Path(self.ws_dir) / "model" / 'matches.h5'
        match_features.main(
            matcher_conf, sampled_pairs_file,
            features=features_h5,
            matches=matches_h5
        )
        print(f"  ✓ Sampled frames matched")

        # Step 5 & 7: Run calibration and match remaining frames in parallel
        print("Step 5 & 7: Running calibration and matching remaining frames in parallel...")

        # Prepare data structures for parallel execution
        calibrated_params = {'completed': False, 'error': None, 'params': None}
        matching_result = {'completed': False, 'error': None, 'pairs': None}

        def run_calibration():
            try:
                params = self._calibrate_with_pycolmap(sampled_names, sampled_pairs, features_h5, matches_h5)
                calibrated_params['params'] = params
                calibrated_params['completed'] = True
                print(f"  ✓ Calibrated intrinsics: fx={params['fx']:.2f}, fy={params['fy']:.2f}, "
                        f"k1={params['k1']:.6f}, k2={params['k2']:.6f}")
            except Exception as e:
                print(f"  ✗ Calibration error: {e}")
                traceback.print_exc()
                calibrated_params['error'] = e

        def run_remaining_matching():
            try:
                sfm_pairs = Path(f'{self.ws_dir}/pairs-sfm.txt')
                filtered_pairs = self._generate_pairs_excluding_sampled(sampled_names, sfm_pairs)
                print(f"  ✓ Generated {len(filtered_pairs)} pairs for remaining frames")

                match_features.main(
                    matcher_conf, sfm_pairs,
                    features=features_h5,
                    matches=matches_h5
                )
                matching_result['pairs'] = filtered_pairs
                matching_result['completed'] = True
                print(f"  ✓ Remaining frames matched")
            except Exception as e:
                print(f"  ✗ Matching error: {e}")
                traceback.print_exc()
                matching_result['error'] = e

        # Start both threads
        calibration_thread = threading.Thread(target=run_calibration, name="CalibrationThread")
        matching_thread = threading.Thread(target=run_remaining_matching, name="MatchingThread")

        calibration_thread.start()
        matching_thread.start()

        # Wait for both to complete
        calibration_thread.join()
        matching_thread.join()

        # Check for errors
        if calibrated_params['error'] is not None:
            raise calibrated_params['error']
        if matching_result['error'] is not None:
            raise matching_result['error']

        params = calibrated_params['params']
        filtered_pairs = matching_result['pairs']

        # Step 6: Update cameras.bin with calibrated intrinsics
        print("Step 6: Updating cameras with calibrated intrinsics...")
        self._update_cameras(params)

        # Step 8: Run triangulation with all frames
        print("Step 8: Running triangulation with all frames...")
        sparse_reconstruction_folder = Path(self.ws_dir) / 'sparse' / '0'
        os.makedirs(sparse_reconstruction_folder, exist_ok=True)

        # Combine all pairs for triangulation
        all_pairs = sampled_pairs + filtered_pairs
        all_pairs_file = Path(self.ws_dir) / 'pairs-all.txt'
        with open(all_pairs_file, 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in all_pairs))
        print(f"  Total pairs for triangulation: {len(all_pairs)}")

        triangulation.main(
            sparse_reconstruction_folder,
            Path(f'{self.ws_dir}/model'),
            Path(self.images_dir),
            all_pairs_file, features, matches_h5,
            skip_geometric_verification=True
        )

        # Verify outputs exist
        if (sparse_reconstruction_folder / 'points3D.bin').exists():
            print(f"  ✓ points3D.bin created successfully at {sparse_reconstruction_folder}")
        else:
            raise RuntimeError(f"points3D.bin not created at {sparse_reconstruction_folder}")

        # Step 9: Convert final cameras to OPENCV
        print("Step 9: Converting final cameras to OPENCV model...")
        convert_cameras_to_opencv(sparse_reconstruction_folder)

        # Cleanup
        print("Step 10: Cleaning up temporary files...")
        shutil.rmtree(os.path.join(self.ws_dir, 'model'))

        print("="*60)
        print("✓ BUNDLE ADJUSTMENT WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*60)


    def _sample_frames(self, num_samples):
        """Sample frames uniformly from pose_dict."""
        all_names = sorted(self.pose_dict.keys())
        total_frames = len(all_names)

        if total_frames <= num_samples:
            print(f"Warning: Dataset has {total_frames} frames, using all for calibration")
            return all_names

        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        return [all_names[i] for i in indices]

    def _calibrate_with_pycolmap(self, sampled_names, sampled_pairs, features_h5, matches_h5):
        """Run hloc reconstruction on sampled frames to get calibrated intrinsics."""
        import pycolmap
        from gaustudio.utils.colmap_utils import create_cameras_and_points_bin

        # Create temporary workspace
        mapper_ws = Path(self.ws_dir) / 'mapper_calibration'
        mapper_ws.mkdir(exist_ok=True)
        mapper_images = mapper_ws / 'images'
        mapper_model = mapper_ws / 'model'
        mapper_sparse = mapper_ws / 'sparse' / '0'

        mapper_images.mkdir(exist_ok=True)
        mapper_model.mkdir(exist_ok=True)
        mapper_sparse.mkdir(parents=True, exist_ok=True)

        # Copy sampled images
        for img_name in sampled_names:
            src = Path(self.images_dir) / self.filename_lookup[img_name]
            dst = mapper_images / self.filename_lookup[img_name]
            if src.exists():
                shutil.copy2(src, dst)

        # Get intrinsics from original model
        cameras = read_cameras_binary(str(Path(self.ws_dir) / 'model' / 'cameras.bin'))
        orig_cam = list(cameras.values())[0]

        if orig_cam.model == 'PINHOLE':
            fx, fy, cx, cy = orig_cam.params
        elif orig_cam.model == 'SIMPLE_PINHOLE':
            f, cx, cy = orig_cam.params
            fx = fy = f
        elif orig_cam.model == 'SIMPLE_RADIAL':
            f, cx, cy = orig_cam.params[:3]
            fx = fy = f
        elif orig_cam.model == 'RADIAL':
            f, cx, cy = orig_cam.params[:3]
            fx = fy = f
        elif orig_cam.model == 'OPENCV':
            fx, fy, cx, cy = orig_cam.params[:4]
        else:
            raise ValueError(f"Unsupported camera model: {orig_cam.model}")

        intrinsics = {
            'width': orig_cam.width, 'height': orig_cam.height,
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
        }

        # Create COLMAP model for sampled frames
        create_cameras_and_points_bin(str(mapper_model), intrinsics)

        # Create images.bin using create_images_from_pose_dict
        sampled_pose_dict = {name: self.pose_dict[name] for name in sampled_names}
        sampled_filename_lookup = {name: self.filename_lookup[name] for name in sampled_names}
        create_images_from_pose_dict(str(mapper_ws), sampled_pose_dict, sampled_filename_lookup)

        # Write pairs file
        sampled_pairs_file = mapper_ws / 'pairs-sampled.txt'
        with open(sampled_pairs_file, 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in sampled_pairs))

        # Create image list for hloc reconstruction
        image_list = [self.filename_lookup[name] for name in sampled_names]

        # Run hloc reconstruction with bundle adjustment in OPENCV mode
        print("  Running hloc reconstruction with bundle adjustment...")
        try:
            # Specify OPENCV camera model through image_options
            image_options = {
                'camera_model': 'OPENCV'
            }

            rec = reconstruction.main(
                sfm_dir=mapper_sparse,
                image_dir=mapper_images,
                pairs=sampled_pairs_file,
                features=features_h5,
                matches=matches_h5,
                camera_mode=pycolmap.CameraMode.SINGLE,
                verbose=False,
                skip_geometric_verification=True,
                image_list=image_list,
                image_options=image_options,
            )
            print(f"  Hloc reconstruction completed")
        except Exception as e:
            print(f"  Error running hloc reconstruction: {e}")
            raise

        # Read calibrated cameras
        calib_cameras_path = mapper_sparse / 'cameras.bin'
        if not calib_cameras_path.exists():
            print(f"  Error: No reconstruction found in {mapper_sparse}")
            print(f"  Available files: {list(mapper_sparse.glob('*'))}")
            raise RuntimeError("Hloc reconstruction failed to produce calibrated cameras")

        calib_cameras = read_cameras_binary(str(calib_cameras_path))
        cam = list(calib_cameras.values())[0]

        # Extract parameters
        if cam.model == 'OPENCV':
            fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
        elif cam.model == 'PINHOLE':
            fx, fy, cx, cy = cam.params
            k1 = k2 = p1 = p2 = 0.0
        elif cam.model == 'SIMPLE_PINHOLE':
            f, cx, cy = cam.params
            fx = fy = f
            k1 = k2 = p1 = p2 = 0.0
        elif cam.model == 'SIMPLE_RADIAL':
            f, cx, cy, k1 = cam.params
            fx = fy = f
            k2 = p1 = p2 = 0.0
        elif cam.model == 'RADIAL':
            f, cx, cy, k1, k2 = cam.params
            fx = fy = f
            p1 = p2 = 0.0
        else:
            raise ValueError(f"Unsupported camera model: {cam.model}")

        return {
            'width': cam.width, 'height': cam.height,
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2
        }

    def _update_cameras(self, calibrated_params):
        """Update cameras.bin with calibrated intrinsics."""
        cameras_path = Path(self.ws_dir) / 'model' / 'cameras.bin'
        cameras = read_cameras_binary(str(cameras_path))

        updated_cameras = {}
        for cam_id, cam in cameras.items():
            opencv_params = np.array([
                calibrated_params['fx'], calibrated_params['fy'],
                calibrated_params['cx'], calibrated_params['cy'],
                calibrated_params['k1'], calibrated_params['k2'],
                calibrated_params['p1'], calibrated_params['p2'],
            ], dtype=np.float64)

            updated_cameras[cam_id] = Camera(
                id=cam.id, model='OPENCV',
                width=calibrated_params['width'],
                height=calibrated_params['height'],
                params=opencv_params
            )

        write_cameras_binary(updated_cameras, str(cameras_path))
        print(f"  Updated {len(updated_cameras)} cameras with calibrated intrinsics")

    def _generate_pairs_for_sampled(self, sampled_names):
        """Generate pairs only for sampled frames."""
        images = read_images_binary(Path(self.ws_dir) / 'model' / 'images.bin')
        # Optimization: use configurable overlap
        overlap = getattr(self, 'pair_overlap', 5)
        all_pairs = pairs_from_poses(images, overlap=overlap)

        # Keep only pairs where both images are in sampled set
        sampled_set = set(sampled_names)
        sampled_pairs = []
        for name1, name2 in all_pairs:
            base1 = name1.replace('.jpg', '').replace('.png', '')
            base2 = name2.replace('.jpg', '').replace('.png', '')
            if base1 in sampled_set and base2 in sampled_set:
                sampled_pairs.append((name1, name2))

        return sampled_pairs

    def _generate_pairs_excluding_sampled(self, sampled_names, output_path):
        """Generate pairs excluding those within sampled frames."""
        images = read_images_binary(Path(self.ws_dir) / 'model' / 'images.bin')
        # Optimization: use configurable overlap
        overlap = getattr(self, 'pair_overlap', 10)
        all_pairs = pairs_from_poses(images, overlap=overlap)

        # Filter out pairs where both images are in sampled set
        sampled_set = set(sampled_names)
        filtered_pairs = []
        for name1, name2 in all_pairs:
            base1 = name1.replace('.jpg', '').replace('.png', '')
            base2 = name2.replace('.jpg', '').replace('.png', '')
            if not (base1 in sampled_set and base2 in sampled_set):
                filtered_pairs.append((name1, name2))

        # Write pairs to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in filtered_pairs))

        return filtered_pairs


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
