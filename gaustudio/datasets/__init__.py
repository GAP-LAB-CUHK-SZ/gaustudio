datasets = {}
import numpy as np
import cv2

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    
@dataclasses.dataclass
class Camera(nn.Module):

    R: np.ndarray
    T: np.ndarray 
    FoVx: float
    FoVy: float
    
    image_width: int
    image_height: int
    
    znear: float = 0.1
    zfar: float = 100
    
    trans: np.array = np.array([0.0, 0.0, 0.0])
    scale: float = 1.0

    world_view_transform: torch.Tensor = None 
    full_proj_transform: torch.Tensor = None

    image_path: str = None
    image: np.array = None

    def __post_init__(self):
        self._setup()
    
    def _setup(self):
        if self.world_view_transform is None:
            self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1)
        if self.full_proj_transform is None:
            projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1)
            self.full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        if self.image_path is not None:
            self.image = Image.open(image_path)
            self.image_height, self.image_width, _ = image.shape
            
        # Compute camera center from inverse view matrix
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
    
    @property
    def extrinsics(self):
        extrinsic = np.hstack((self.R, self.T.reshape(-1, 1)))
        extrinsic = np.vstack((extrinsic,[0,0,0,1]))
        return torch.from_numpy(extrinsic).float().to(self.data_device)
        
    @extrinsics.setter
    def extrinsics(self, value):
        """Sets the extrinsic parameters of the camera"""

        # The rotation matrix R is the top left 3x3 block of the extrinsics
        R = extrinsics[:3, :3]
        
        # The translation vector T is the top right 3x1 block of the extrinsics
        T = extrinsics[:3, 3]

        self.R = R
        self.T = T

        # Update the world view transform and full projection transform after setting new extrinsics
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    @staticmethod
    def from_colmap(cam_info):
        return Camera(
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX, 
            FoVy=cam_info.FovY
        )

def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, config):
    dataset = datasets[name](config)
    return dataset


from . import colmap