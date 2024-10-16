import rerun as rr
from pathlib import Path
from typing import Literal
from typing import Optional, TypedDict, Union, List, Tuple
import copy
import torch
import numpy as np
from jaxtyping import Float32, Bool
import trimesh
from tqdm import tqdm

from mini_dust3r.utils.image import load_images, ImageDict
from mini_dust3r.inference import inference, Dust3rResult
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.image_pairs import make_pairs
from mini_dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from mini_dust3r.cloud_opt.base_opt import BasePCOptimizer
from mini_dust3r.viz import pts3d_to_trimesh, cat_meshes
from dataclasses import dataclass


@dataclass
class OptimizedResult:
    K_b33: Float32[np.ndarray, "b 3 3"]
    world_T_cam_b44: Float32[np.ndarray, "b 4 4"]
    rgb_hw3_list: list[Float32[np.ndarray, "h w 3"]]
    depth_hw_list: list[Float32[np.ndarray, "h w"]]
    conf_hw_list: list[Float32[np.ndarray, "h w"]]
    masks_list: Bool[np.ndarray, "h w"]
    point_cloud: trimesh.PointCloud
    mesh: trimesh.Trimesh


def log_optimized_result(
    optimized_result: OptimizedResult, parent_log_path: Path
) -> None:
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, timeless=True)
    # log pointcloud
    rr.log(
        f"{parent_log_path}/pointcloud",
        rr.Points3D(
            positions=optimized_result.point_cloud.vertices,
            colors=optimized_result.point_cloud.colors,
        ),
        timeless=True,
    )

    mesh = optimized_result.mesh
    rr.log(
        f"{parent_log_path}/mesh",
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            vertex_colors=mesh.visual.vertex_colors,
            indices=mesh.faces,
        ),
        timeless=True,
    )
    pbar = tqdm(
        zip(
            optimized_result.rgb_hw3_list,
            optimized_result.depth_hw_list,
            optimized_result.K_b33,
            optimized_result.world_T_cam_b44,
        ),
        total=len(optimized_result.rgb_hw3_list),
    )
    for i, (rgb_hw3, depth_hw, k_33, world_T_cam_44) in enumerate(pbar):
        camera_log_path = f"{parent_log_path}/camera_{i}"
        height, width, _ = rgb_hw3.shape
        rr.log(
            f"{camera_log_path}",
            rr.Transform3D(
                translation=world_T_cam_44[:3, 3],
                mat3x3=world_T_cam_44[:3, :3],
                from_parent=False,
            ),
        )
        rr.log(
            f"{camera_log_path}/pinhole",
            rr.Pinhole(
                image_from_camera=k_33,
                height=height,
                width=width,
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )
        rr.log(
            f"{camera_log_path}/pinhole/rgb",
            rr.Image(rgb_hw3),
        )
        rr.log(
            f"{camera_log_path}/pinhole/depth",
            rr.DepthImage(depth_hw),
        )


def scene_to_results(scene: BasePCOptimizer, min_conf_thr: int) -> OptimizedResult:
    ### get camera parameters K and T
    K_b33: Float32[np.ndarray, "b 3 3"] = scene.get_intrinsics().numpy(force=True)
    world_T_cam_b44: Float32[np.ndarray, "b 4 4"] = scene.get_im_poses().numpy(
        force=True
    )
    ### image, confidence, depths
    rgb_hw3_list: list[Float32[np.ndarray, "h w 3"]] = scene.imgs
    depth_hw_list: list[Float32[np.ndarray, "h w"]] = [
        depth.numpy(force=True) for depth in scene.get_depthmaps()
    ]
    # normalized depth
    # depth_hw_list = [depth_hw / depth_hw.max() for depth_hw in depth_hw_list]

    conf_hw_list: list[Float32[np.ndarray, "h w"]] = [
        c.numpy(force=True) for c in scene.im_conf
    ]
    # normalize confidence
    # conf_hw_list = [conf_hw / conf_hw.max() for conf_hw in conf_hw_list]

    # point cloud, mesh
    pts3d_list: list[Float32[np.ndarray, "h w 3"]] = [
        pt3d.numpy(force=True) for pt3d in scene.get_pts3d()
    ]
    # get log confidence
    log_conf_trf: Float32[torch.Tensor, ""] = scene.conf_trf(torch.tensor(min_conf_thr))
    # set the minimum confidence threshold
    scene.min_conf_thr = float(log_conf_trf)
    masks_list: Bool[np.ndarray, "h w"] = [
        mask.numpy(force=True) for mask in scene.get_masks()
    ]

    point_cloud: Float32[np.ndarray, "num_points 3"] = np.concatenate(
        [p[m] for p, m in zip(pts3d_list, masks_list)]
    )
    colors: Float32[np.ndarray, "num_points 3"] = np.concatenate(
        [p[m] for p, m in zip(rgb_hw3_list, masks_list)]
    )
    point_cloud = trimesh.PointCloud(
        point_cloud.reshape(-1, 3), colors=colors.reshape(-1, 3)
    )

    meshes = []
    pbar = tqdm(zip(rgb_hw3_list, pts3d_list, masks_list), total=len(rgb_hw3_list))
    for rgb_hw3, pts3d, mask in pbar:
        meshes.append(pts3d_to_trimesh(rgb_hw3, pts3d, mask))

    mesh = trimesh.Trimesh(**cat_meshes(meshes))
    optimised_result = OptimizedResult(
        K_b33=K_b33,
        world_T_cam_b44=world_T_cam_b44,
        rgb_hw3_list=rgb_hw3_list,
        depth_hw_list=depth_hw_list,
        conf_hw_list=conf_hw_list,
        masks_list=masks_list,
        point_cloud=point_cloud,
        mesh=mesh,
    )
    return optimised_result


def inferece_dust3r(
    image_dir_or_list: Union[Path , list[Path]],
    model: AsymmetricCroCo3DStereo,
    device: Literal["cpu", "cuda", "mps"],
    batch_size: int = 1,
    image_size: Literal[224, 512] = 512,
    niter: int = 100,
    schedule: Literal["linear", "cosine"] = "linear",
    min_conf_thr: float = 10,
) -> OptimizedResult:
    """
    Perform inference using the Dust3r algorithm.

    Args:
        image_dir_or_list (Union[Path, List[Path]]): Path to the directory containing images or a list of image paths.
        model (AsymmetricCroCo3DStereo): The Dust3r model to use for inference.
        device (Literal["cpu", "cuda", "mps"]): The device to use for inference ("cpu", "cuda", or "mps").
        batch_size (int, optional): The batch size for inference. Defaults to 1.
        image_size (Literal[224, 512], optional): The size of the input images. Defaults to 512.
        niter (int, optional): The number of iterations for the global alignment optimization. Defaults to 100.
        schedule (Literal["linear", "cosine"], optional): The learning rate schedule for the global alignment optimization. Defaults to "linear".
        min_conf_thr (float, optional): The minimum confidence threshold for the optimized result. Defaults to 10.

    Returns:
        OptimizedResult: The optimized result containing the RGB, depth, and confidence images.

    Raises:
        ValueError: If `image_dir_or_list` is neither a list of paths nor a path.
    """
    if isinstance(image_dir_or_list, list):
        imgs: list[ImageDict] = load_images(
            folder_or_list=image_dir_or_list, size=image_size, verbose=True
        )
    elif isinstance(image_dir_or_list, Path):
        imgs: list[ImageDict] = load_images(
            folder_or_list=str(image_dir_or_list), size=image_size, verbose=True
        )
    else:
        raise ValueError("image_dir_or_list should be a list of paths or a path")

    # if only one image was loaded, duplicate it to feed into stereo network
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1

    pairs: list[tuple[ImageDict, ImageDict]] = make_pairs(
        imgs, scene_graph="swin", prefilter=None, symmetrize=True
    )
    output: Dust3rResult = inference(pairs, model, device, batch_size=batch_size)

    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(imgs) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene: BasePCOptimizer = global_aligner(
        dust3r_output=output, device=device, mode=mode
    )

    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )

    # get the optimized result from the scene
    optimized_result: OptimizedResult = scene_to_results(scene, min_conf_thr)
    return optimized_result
