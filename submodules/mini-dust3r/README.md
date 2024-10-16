# Mini-Dust3r
A miniature version of [dust3r](https://github.com/naver/dust3r) only for performing inference.
This makes it much easier to use without needing the training/data/eval code. Tested on Linux, Apple Silicon Macs, and Windows (Thanks @Vincentqyw)
<p align="center">
  <img src="media/mini-dust3r.gif" alt="example output" width="720" />
</p>


## Installation
Easily installable via pip
```bash
pip install mini-dust3r
```

## Demo
A hosted demo can be found on huggingface here <a href='https://huggingface.co/spaces/pablovela5620/mini-dust3r'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

or from source using [Pixi](http://pixi.sh)

``` bash
git clone https://github.com/pablovela5620/mini-dust3r.git
pixi run gradio-demo
```

You can also just use rerun demo directly with
```bash
pixi run rerun-demo
```

## Minimal Example
Uses [Rerun](http://rerun.io/) to visualize the outputs

```python
import rerun as rr
from pathlib import Path
from argparse import ArgumentParser
import torch

from mini_dust3r.api import OptimizedResult, inferece_dust3r, log_optimized_result
from mini_dust3r.model import AsymmetricCroCo3DStereo


def main(image_dir: Path):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    optimized_results: OptimizedResult = inferece_dust3r(
        image_dir_or_list=image_dir,
        model=model,
        device=device,
        batch_size=1,
    )
    log_optimized_result(optimized_results, Path("world"))


if __name__ == "__main__":
    parser = ArgumentParser("mini-dust3r rerun demo script")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to process",
        required=True,
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "mini-dust3r")
    main(args.image_dir)
    rr.script_teardown(args)
```

## Inputs and Outputs

### Inference Fuction

```python
def inferece_dust3r(
    image_dir_or_list: Path | list[Path],
    model: AsymmetricCroCo3DStereo,
    device: Literal["cpu", "cuda", "mps"],
    batch_size: int = 1,
    image_size: Literal[224, 512] = 512,
    niter: int = 100,
    schedule: Literal["linear", "cosine"] = "linear",
    min_conf_thr: float = 10,
) -> OptimizedResult:
```
Consists of
* image_dir_or_list - Path to the directory containing images or a list of image paths
* model - The Dust3r model to use for inference
* device - device to use for inference ("cpu", "cuda", or "mps")
* batch_size - The batch size for inference. Defaults to 1.
* image_size - The size of the input images. Defaults to 512.
* niter - The number of iterations for the global alignment optimization. Defaults to 100.
* schedule - The learning rate schedule for the global alignment optimization. Defaults to "linear"
* min_conf_thr - The minimum confidence threshold for the optimized result. Defaults to 10.

### Output from OptimizedResult

```python
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
```
Consists of
* K_b33 - camera intrinsics of shape (b33)
* world_T_cam_b44 - camera to world transformation matrix of shape b44
     in OpenCV convention X - Right Y - Down Z - Forward (RDF)
* rgb_hw3_list - list of RGB images shape (list[hw3])
* depth_hw_list - list of normalized depth maps shape (list[hw])
* conf_hw_list - list of normalized confidence values (list[hw])
* mask_list - list of masks (list[hw])
* point cloud - as a trimesh pointcloud object
* mesh - as a trimesh mesh object

## References
Full credit goes the Naver for their great work on
* [Dust3r](https://github.com/naver/dust3r)
* [Croco](https://github.com/naver/croco)
