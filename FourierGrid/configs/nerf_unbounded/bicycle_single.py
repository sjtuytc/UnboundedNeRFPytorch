_base_ = './nerf_unbounded_default.py'
expname = 'bicycle_nov29_'
vis = dict(
    height_rate = 0.6 # camera direction frustrum height
)
model='FourierGrid'
basedir = './logs/360'
alpha_init = 1e-4
stepsize = 0.5
_mpi_depth = 256
coarse_iter = 0
# fast_color_thres={   # default
#         '_delete_': True,                           # to ignore the base config
#         0   : alpha_init*stepsize/10,               # 0.5e-5
#         1500: min(alpha_init, 1e-4)*stepsize/5,     # 1e-5
#         2500: min(alpha_init, 1e-4)*stepsize/2,     # 2.5e-5
#         3500: min(alpha_init, 1e-4)*stepsize/1.5,   
#         4500: min(alpha_init, 1e-4)*stepsize,
#         5500: min(alpha_init, 1e-4),
#         6500: 1e-4,
#     }
fast_color_thres = 1e-4
maskout_near_cam_vox = False
pervoxel_lr = False
weight_distortion = 0.05
data = dict(
    dataset_type='llff',
    datadir='./data/360_v2/bicycle',
    factor=16, # 1237x822
    # width=320,
    # height=240,
    movie_render_kwargs=dict(
        shift_x=0.0,  # positive right
        shift_y=0, # negative down
        shift_z=0,
        scale_r=1.0,
        pitch_deg=-10, # negative look downward
    ),
)

coarse_train = dict(N_iters=0)

fine_train = dict(
    N_iters=100000,
    N_rand=2048,
    lrate_decay=80,
    ray_sampler='flatten',
    weight_nearclip=1.0,
    weight_distortion=weight_distortion,
    # pg_scale=[2000,4000,6000,8000,10000,12000,14000,16000],
    pg_scale=[1000, 2000, 3000, 4000, 5000, 6000, 7000],
    tv_before=20000,
    tv_dense_before=20000,
    weight_tv_density=1e-6,
    weight_tv_k0=1e-7,
    weight_main=1.0,
    weight_freq=5.0,
)

voxel_size_for_all = 200 # default 220
voxel_size_density = voxel_size_for_all
voxel_size_rgb = voxel_size_for_all
voxel_size_viewdir = -1

fine_model_and_render = dict(
    num_voxels_density=voxel_size_density**3,
    num_voxels_base_density=voxel_size_density**3,
    num_voxels_rgb=voxel_size_rgb**3,
    num_voxels_base_rgb=voxel_size_rgb**3,
    num_voxels_viewdir=voxel_size_viewdir**3,
    alpha_init=alpha_init,
    stepsize=stepsize,
    fast_color_thres=fast_color_thres,
    world_bound_scale=1,
    rgbnet_dim=12, # default
    rgbnet_depth=3, # default
    bbox_thres=-1,
    maskout_near_cam_vox=False,
    bg_len=0.2,
)