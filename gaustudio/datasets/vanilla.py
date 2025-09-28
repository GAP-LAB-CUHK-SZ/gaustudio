import json
from gaustudio import datasets
from gaustudio.datasets.base import BaseDataset
from gaustudio.datasets.utils import JSON_to_camera
from typing import Dict

class VanillaDatasetBase(BaseDataset):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.image_path = self.source_path / "images"
        
        self._initialize()
        self.ply_path = None
        
    def _initialize(self):
        print("Loading vanilla dataset cameras...")
        with open(self.source_path / f"cameras.json", 'r') as f:
            camera_data = json.load(f)

        print(f"Processing {len(camera_data)} vanilla cameras...")

        def process_camera(camera_dict):
            """Process a single camera with optimizations"""
            try:
                _camera = JSON_to_camera(camera_dict, "cuda")
                _image_path = self.image_path / camera_dict['img_name']

                # Check if image exists before processing
                if not _image_path.exists():
                    print(f"Warning: Image not found: {_image_path}")
                    return None

                _camera.load_image(_image_path)
                return _camera
            except Exception as e:
                print(f"Error processing camera {camera_dict.get('img_name', 'unknown')}: {e}")
                return None

        all_cameras = self.process_in_parallel(
            camera_data,
            process_camera,
            desc="Processing vanilla cameras",
        )

        print(f"Successfully processed {len(all_cameras)} cameras")
        self.finalize_cameras(all_cameras)
            
@datasets.register('vanilla')
class VanillaDataset(VanillaDatasetBase):
    pass
