iterations = 30_000

position_lr_init = 0.00016
position_lr_final = 0.0000016
position_lr_delay_mult = 0.01
position_lr_max_steps = 30_000
feature_lr = 0.0025
opacity_lr = 0.05
scaling_lr = 0.005
rotation_lr = 0.001
percent_dense = 0.01

lambda_dssim = 0.2
lambda_depth = 0.
lambda_depth_correlation = 0.
lambda_depth_consistency = 0.
depth_from_iter = 2000
depth_until_iter = 30_000
lambda_patches = 0.03
patches_num = 10
patches_size = 256
lambda_scale = 0.
scale_from_iter = 15_000
scale_until_iter = 30_000
lambda_entropy = 0.
entropy_from_iter = 7_000
entropy_until_iter = 30_000
densification_interval = 100
opacity_reset_interval = 3000
densify_from_iter = 500
densify_until_iter = 15_000
densify_grad_threshold = 0.0002

random_background = False
with_background = False