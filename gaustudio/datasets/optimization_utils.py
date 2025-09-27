"""
Optimization utilities for dataset loading and processing.
Common performance patterns used across all dataset implementations.
"""

import os
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Callable, Any, Optional


class OptimizedImageLoader:
    """Optimized image loading with caching and error handling"""

    @staticmethod
    def load_image_optimized(image_path: str, target_dtype: np.dtype = np.float32) -> Optional[torch.Tensor]:
        """
        Load and convert image with optimizations

        Args:
            image_path: Path to the image file
            target_dtype: Target numpy dtype for conversion

        Returns:
            Tensor image in RGB format, normalized to [0, 1], or None if failed
        """
        try:
            if not os.path.exists(image_path):
                return None

            # Load image with error checking
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Optimized color conversion and normalization
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb.astype(target_dtype) / 255.0)

            return image_tensor
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    @staticmethod
    def load_depth_optimized(depth_path: str, scale_factor: float = 1000.0) -> Optional[torch.Tensor]:
        """
        Load depth image with optimizations

        Args:
            depth_path: Path to the depth file
            scale_factor: Scale factor for depth values

        Returns:
            Depth tensor or None if failed
        """
        try:
            if not os.path.exists(depth_path):
                return None

            # Use optimized flags for depth loading
            depth_data = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            if depth_data is None:
                return None

            # Convert to float tensor with scaling
            depth_tensor = torch.from_numpy(depth_data.astype(np.float32) / scale_factor)
            return depth_tensor
        except Exception as e:
            print(f"Error loading depth {depth_path}: {e}")
            return None

    @staticmethod
    def load_mask_optimized(mask_path: str, target_size: tuple) -> Optional[torch.Tensor]:
        """
        Load and process mask with optimizations

        Args:
            mask_path: Path to the mask file
            target_size: Target size (width, height)

        Returns:
            Mask tensor or None if failed
        """
        try:
            if not os.path.exists(mask_path):
                return None

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None

            # Optimized thresholding and resizing
            mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
            if mask.shape[:2] != target_size[::-1]:  # cv2 uses (height, width)
                mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

            mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0)
            return mask_tensor
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return None


class ParallelProcessor:
    """Utilities for parallel processing of dataset elements"""

    @staticmethod
    def process_parallel(
        items: List[Any],
        process_func: Callable,
        max_workers: Optional[int] = None,
        desc: str = "Processing"
    ) -> List[Any]:
        """
        Process items in parallel with progress tracking

        Args:
            items: List of items to process
            process_func: Function to apply to each item
            max_workers: Maximum number of worker threads
            desc: Description for progress bar

        Returns:
            List of processed results (None entries filtered out)
        """
        if max_workers is None:
            max_workers = min(8, len(items), os.cpu_count())  # Optimal for I/O bound operations

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_func, item) for item in items]

            for future in tqdm(futures, desc=desc):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error in parallel processing: {e}")

        return results


class VectorizedOperations:
    """Optimized vectorized operations for common dataset computations"""

    @staticmethod
    def batch_transform_matrices(transform_matrices: np.ndarray) -> tuple:
        """
        Vectorized processing of transformation matrices

        Args:
            transform_matrices: Array of shape (N, 4, 4)

        Returns:
            Tuple of (R_batch, T_batch) with optimized transformations
        """
        # Apply common transformations in vectorized form
        transform_matrices = transform_matrices.astype(np.float32)

        # Common coordinate system adjustments
        transform_matrices[:, :, 1:3] *= -1

        # Batch inverse calculation
        extrinsics_batch = np.linalg.inv(transform_matrices)

        # Extract rotation and translation
        R_batch = np.transpose(extrinsics_batch[:, :3, :3], (0, 2, 1))
        T_batch = extrinsics_batch[:, :3, 3]

        return R_batch, T_batch

    @staticmethod
    def batch_fov_calculation(focal_lengths: np.ndarray, image_sizes: np.ndarray) -> np.ndarray:
        """
        Vectorized field of view calculation

        Args:
            focal_lengths: Array of focal lengths
            image_sizes: Array of image sizes

        Returns:
            Array of FoV values
        """
        return 2.0 * np.arctan(image_sizes / (2.0 * focal_lengths))


class MemoryOptimizer:
    """Memory optimization utilities"""

    @staticmethod
    def optimize_tensor_dtype(tensor: torch.Tensor, preserve_precision: bool = True) -> torch.Tensor:
        """
        Optimize tensor memory usage while preserving precision

        Args:
            tensor: Input tensor
            preserve_precision: Whether to preserve float32 precision

        Returns:
            Memory-optimized tensor
        """
        if tensor.dtype == torch.float64 and not preserve_precision:
            return tensor.float()  # Convert to float32
        elif tensor.dtype == torch.uint8:
            return tensor  # Already optimized
        return tensor

    @staticmethod
    def batch_process_with_memory_limit(
        items: List[Any],
        process_func: Callable,
        batch_size: int = 32
    ) -> List[Any]:
        """
        Process items in batches to control memory usage

        Args:
            items: Items to process
            process_func: Processing function
            batch_size: Size of each batch

        Returns:
            List of processed results
        """
        results = []
        for i in tqdm(range(0, len(items), batch_size), desc="Batch processing"):
            batch = items[i:i + batch_size]
            batch_results = [process_func(item) for item in batch]
            results.extend([r for r in batch_results if r is not None])

        return results


class CacheManager:
    """Simple caching utilities for repeated operations"""

    def __init__(self):
        self._cache = {}

    def get_or_compute(self, key: str, compute_func: Callable) -> Any:
        """
        Get cached value or compute and cache it

        Args:
            key: Cache key
            compute_func: Function to compute value if not cached

        Returns:
            Cached or computed value
        """
        if key not in self._cache:
            self._cache[key] = compute_func()
        return self._cache[key]

    def clear(self):
        """Clear the cache"""
        self._cache.clear()


# Global cache manager instance
global_cache = CacheManager()


def apply_common_optimizations(dataset_class):
    """
    Decorator to apply common optimizations to dataset classes

    Usage:
        @apply_common_optimizations
        class MyDataset(Dataset):
            ...
    """
    original_init = dataset_class.__init__

    def optimized_init(self, *args, **kwargs):
        # Initialize with memory optimization flags
        self._memory_optimizer = MemoryOptimizer()
        self._cache_manager = CacheManager()

        # Call original init
        original_init(self, *args, **kwargs)

        # Apply post-initialization optimizations
        if hasattr(self, 'all_cameras'):
            print(f"Optimizing {len(self.all_cameras)} cameras for memory usage...")
            for camera in self.all_cameras:
                if hasattr(camera, 'image') and camera.image is not None:
                    camera.image = self._memory_optimizer.optimize_tensor_dtype(camera.image)
                if hasattr(camera, 'depth') and camera.depth is not None:
                    camera.depth = self._memory_optimizer.optimize_tensor_dtype(camera.depth)

    dataset_class.__init__ = optimized_init
    return dataset_class