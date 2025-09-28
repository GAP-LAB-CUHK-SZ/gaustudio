import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import getNerfppNorm, camera_to_JSON


CameraSortKey = Callable[[object], object]


class BaseDataset(Dataset):
    """Shared functionality for datasets that load collections of cameras."""

    required_keys: Sequence[str] = ("source_path",)

    @staticmethod
    def default_sort_key(camera: object) -> object:
        return getattr(camera, "image_name", "")

    def __init__(self, config):
        super().__init__()
        self.config = dict(config) if config is not None else {}
        self._validate_config()
        self.source_path = Path(self.config["source_path"]) if "source_path" in self.config else None
        self.white_background = self.config.get("white_background", False)
        self.all_cameras = []
        self.nerf_normalization = {}
        self.cameras_extent = None
        self.cameras_center = None
        self.cameras_min_extent = None
        parallel_config = self.config.get("parallel_loading", True)
        if isinstance(parallel_config, str):
            parallel_config = parallel_config.strip().lower() not in {"0", "false", "no", "off"}
        else:
            parallel_config = bool(parallel_config)
        self.parallel_loading = parallel_config
        self.num_workers = self._resolve_num_workers(self.config.get("num_workers"))

    def _resolve_num_workers(self, requested: Optional[Any] = None) -> int:
        """Return a safe worker count based on the requested value and CPU limits."""
        cpu_count = os.cpu_count() or 1
        default_workers = min(8, cpu_count)

        if requested is None:
            return default_workers

        try:
            workers = int(requested)
        except (TypeError, ValueError):
            return default_workers

        if workers <= 0:
            return 1

        return min(workers, cpu_count)

    def process_in_parallel(
        self,
        items: Iterable[Any],
        process_fn: Callable[[Any], Any],
        *,
        desc: Optional[str] = None,
        max_workers: Optional[Any] = None,
        filter_none: bool = True,
    ) -> List[Any]:
        """Process items in parallel using a thread pool, filtering ``None`` results by default."""

        item_list = list(items)
        if not item_list:
            return []

        if max_workers is None:
            worker_count = self.num_workers
        else:
            worker_count = self._resolve_num_workers(max_workers)

        worker_count = max(1, min(worker_count, len(item_list)))

        if not self.parallel_loading or worker_count <= 1:
            progress = tqdm(item_list, desc=desc) if desc else None
            iterable = progress if progress is not None else item_list
            results = []
            for item in iterable:
                result = process_fn(item)
                if filter_none and result is None:
                    continue
                results.append(result)
            if progress is not None:
                progress.close()
            return results

        ordered_results: List[Optional[Any]] = [None] * len(item_list)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                (idx, executor.submit(process_fn, item))
                for idx, item in enumerate(item_list)
            ]

            progress = tqdm(total=len(futures), desc=desc) if desc else None
            for idx, future in futures:
                ordered_results[idx] = future.result()
                if progress is not None:
                    progress.update(1)
            if progress is not None:
                progress.close()

        if filter_none:
            return [result for result in ordered_results if result is not None]
        return ordered_results

    def _validate_config(self):
        missing = [key for key in self.required_keys if key not in self.config]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Config must contain keys: {joined}")

    def finalize_cameras(self, cameras: Iterable, sort_key: Optional[CameraSortKey] = None):
        valid = [camera for camera in cameras if camera is not None]
        if not valid:
            raise ValueError("No valid cameras were produced by dataset loader")

        key = sort_key or self.default_sort_key
        if key is not None:
            valid = sorted(valid, key=key)

        self.all_cameras = valid
        self._update_normalization()

    def _update_normalization(self):
        if not self.all_cameras:
            self.nerf_normalization = {}
            self.cameras_extent = None
            self.cameras_center = None
            self.cameras_min_extent = None
            return

        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]
        self.cameras_center = self.nerf_normalization["translate"]
        self.cameras_min_extent = self.nerf_normalization["min_radius"]

    def downsample_scale(self, scale):
        self.all_cameras = [camera.downsample_scale(scale) for camera in self.all_cameras]
        self._update_normalization()
        return self

    def export(self, save_path):
        json_cams = [camera_to_JSON(idx, cam) for idx, cam in enumerate(self.all_cameras)]
        with open(save_path, "w") as file:
            json.dump(json_cams, file)

    def __len__(self):
        return len(self.all_cameras)

    def __getitem__(self, index):
        return self.all_cameras[index]
