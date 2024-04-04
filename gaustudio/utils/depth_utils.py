import cv2
import torch
import numpy as np

def compute_scale_and_shift_ls(prediction, target, mask):
    # tuple specifying with axes to sum
    sum_axes = (0, 1)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, sum_axes)
    a_01 = np.sum(mask * prediction, sum_axes)
    a_11 = np.sum(mask, sum_axes)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, sum_axes)
    b_1 = np.sum(mask * target, sum_axes)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def apply_depth_colormap(depth: np.ndarray, near_plane=None, far_plane=None): 
    near_plane = near_plane or np.min(depth)
    far_plane = far_plane or np.max(depth)
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = np.clip(depth, 0, 1)
    import matplotlib.pyplot as plt
    colored_image = plt.cm.viridis(depth)[:, :, :3]

    return colored_image


class LeastSquaresDepthEstimator(object):
    def __init__(self):
        self._model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to("cuda:0").eval()
        self._transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    
    def predict_depth_single(self, input_image, device="cuda:0"):
        input_height, input_width = np.shape(input_image)[0], np.shape(input_image)[1]
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_batch = self._transforms(input_image).to(device)

        with torch.no_grad():
            depth_pred = self._model(input_batch)

            depth_pred = (
                    torch.nn.functional.interpolate(
                        depth_pred.unsqueeze(1),
                        size=(input_height, input_width),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )
        return depth_pred

    def forward(self, input_image, target_depth, target_depth_mask):
        depth_pred = predict_depth_single(input_image, depth)
        
        depth_scale, depth_shift = compute_scale_and_shift_ls(estimate, target, valid)

        depth_out = depth_pred * depth_scale + depth_shift
        return depth_out.astype(np.float32)