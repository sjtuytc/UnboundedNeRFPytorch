_base_ = '../default.py'
expname = 'oct22_dvgo_Playground_unbounded_baseline'
vis = dict(
    height_rate = 0.6 # camera direction frustrum height
)
basedir = './logs/tanks_and_temple_unbounded'
visualize_poses = False
alpha_init = 1e-4
stepsize = 0.5
_mpi_depth = 256
if visualize_poses:  # for debugging
    unbounded_inward = True
    coarse_iter = 3000
    fast_color_thres=stepsize/_mpi_depth/5
    maskout_near_cam_vox = False
    pervoxel_lr = False
    weight_distortion = 0.0
else:
    unbounded_inward = True
    coarse_iter = 0
    fast_color_thres={
            '_delete_': True,
            0   : alpha_init*stepsize/10,
            1500: min(alpha_init, 1e-4)*stepsize/5,
            2500: min(alpha_init, 1e-4)*stepsize/2,
            3500: min(alpha_init, 1e-4)*stepsize/1.5,
            4500: min(alpha_init, 1e-4)*stepsize,
            5500: min(alpha_init, 1e-4),
            6500: 1e-4,
        }
    maskout_near_cam_vox = False
    pervoxel_lr = False
    # weight_distortion = 1.0
    # weight_distortion = 0.01
    weight_distortion = -1

data = dict(
    dataset_type='nerfpp',
    inverse_y=True,
    white_bkgd=True,
    rand_bkgd=True,
    unbounded_inward=unbounded_inward,
    load2gpu_on_the_fly=True,
    datadir='./data/tanks_and_temples/tat_intermediate_Playground',
    unbounded_inner_r=0.8,
    ndc=False,
)

coarse_train = dict(
    N_iters=coarse_iter,
    pervoxel_lr = pervoxel_lr,
)

fine_train = dict(
    N_iters=40000,
    N_rand=4096,
    ray_sampler='flatten',
    weight_distortion=weight_distortion,
    pg_scale=[1000,2000,3000,4000,5000,6000,7000],
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-6,
    weight_tv_k0=1e-7,
    pervoxel_lr=True,
)

coarse_model_and_render = dict(
    maskout_near_cam_vox = maskout_near_cam_vox,
)

voxel_size = 400  # default 320
# voxel_size = 320  # default 320
fine_model_and_render = dict(
    num_voxels=voxel_size**3,
    num_voxels_base=voxel_size**3,
    alpha_init=alpha_init,
    stepsize=stepsize,
    fast_color_thres=fast_color_thres,
    world_bound_scale=1,
    # contracted_norm='l2', # default
    rgbnet_dim=12, # default
    rgbnet_depth=3, # default
    viewbase_pe=2, # default=4
    bbox_thres=-1,
    maskout_near_cam_vox=False,
)
