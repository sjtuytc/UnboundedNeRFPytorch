_base_ = './nerf_studio_default.py'
expname = 'stump_mar16_'
vis = dict(
    height_rate = 0.6 # camera direction frustrum height
)
model='FourierGrid'
basedir = './logs/stump'
alpha_init = 1e-4
stepsize = 0.5
_mpi_depth = 256
coarse_iter = 0
fast_color_thres={   # default
        '_delete_': True,                           # to ignore the base config
        0   : alpha_init*stepsize/10,               # 0.5e-5
        1500: min(alpha_init, 1e-4)*stepsize/5,     # 1e-5
        2500: min(alpha_init, 1e-4)*stepsize/2,     # 2.5e-5
        3500: min(alpha_init, 1e-4)*stepsize/1.5,   
        4500: min(alpha_init, 1e-4)*stepsize,
        5500: min(alpha_init, 1e-4),
        6500: 1e-4,
    }
maskout_near_cam_vox = False
pervoxel_lr = False
# weight_distortion = 0.01
weight_distortion = 0.02
data = dict(
    dataset_type='nerfstudio',
    spherify=False,  # default: True
    factor=8,
    llffhold=8,
    white_bkgd=True,
    rand_bkgd=True,
    unbounded_inward=True,
    load2gpu_on_the_fly=True,
    bd_factor=None,
    datadir='./data/nerfstudio_data/stump',
    movie_render_kwargs=dict(
        shift_x=0.0,  # positive right
        shift_y=-0.3, # negative down
        shift_z=0,
        scale_r=0.2,
        pitch_deg=-40, # negative look downward
    ),
)

coarse_train = dict(N_iters=0)

fine_train = dict(
    # N_iters=10000, # zelin working
    N_iters=100000,  
    N_rand=2048,
    lrate_decay=80,
    ray_sampler='flatten',
    weight_nearclip=1.0,
    weight_distortion=weight_distortion,
    # pg_scale=[2000,4000,6000,8000,10000,12000,14000,16000],
    pg_scale=[1000, 2000, 3000, 4000, 5000, 6000, 7000], # zelin working
    tv_before=20000,
    tv_dense_before=20000,
    weight_tv_density=1e-6,
    weight_tv_k0=1e-7,
    weight_main=1.0,
    # weight_freq=0.1,
)
voxel_size_density = 200  # default 400
voxel_size_rgb = 200  # default 320
voxel_size_viewdir = -1

fine_model_and_render = dict(
    num_voxels_density=voxel_size_density**3,
    num_voxels_base_density=voxel_size_density**3,
    num_voxels_rgb=voxel_size_rgb**3,
    num_voxels_base_rgb=voxel_size_rgb**3,
    num_voxels_viewdir=voxel_size_viewdir**3,
    alpha_init=alpha_init,
    stepsize=stepsize,
    fast_color_thres={
        '_delete_': True,
        0   : alpha_init*stepsize/10,
        1500: min(alpha_init, 1e-4)*stepsize/5,
        2500: min(alpha_init, 1e-4)*stepsize/2,
        3500: min(alpha_init, 1e-4)*stepsize/1.5,
        4500: min(alpha_init, 1e-4)*stepsize,
        5500: min(alpha_init, 1e-4),
        6500: 1e-4,
    },
    world_bound_scale=1,
)
