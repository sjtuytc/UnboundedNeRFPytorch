_base_ = '../default.py'
data_name = 'building'
model='FourierGrid'
basedir = f'./logs/mega/{data_name}'
visualize_poses = False
alpha_init = 1e-4
stepsize = 0.5
_mpi_depth = 256
maskout_near_cam_vox = False  # changed
pervoxel_lr = False
unbounded_inward = True
expname = f'oct9_mega_{data_name}'
if visualize_poses:  # for debugging only
    coarse_iter = 600
    fast_color_thres=stepsize/_mpi_depth/5
    weight_distortion = 0.0
else:
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
    weight_distortion = -1

data = dict(
    dataset_type='mega',
    inverse_y=True,
    white_bkgd=True,     # almost no effect when rand_bkgd=True
    rand_bkgd=True,      # random background
    unbounded_inward=unbounded_inward,
    load2gpu_on_the_fly=True,
    datadir=f'data/oct9_mega/{data_name}',
    factor=2,
    near_clip = 0.1,
    near = 0.1,
    far = 0.01,
    test_rotate_angle=50, # rotate angle in testing phase
    sample_interval=1,
    num_per_block=-1,  # run this num in block
    unbounded_inner_r=1.0,
    boundary_ratio=0.0,
    # training_ids=['000517', '000520', '000524', ],
)

nerf_em = dict(
    sample_num = 3,
    pos_x_range = 0.1,
    pos_y_range = 0.1,
    pos_z_range = 0.1,
)

coarse_train = dict(
    N_iters=coarse_iter,
    pervoxel_lr = pervoxel_lr,
    ray_sampler='flatten',
)

fine_train = dict(
    N_iters_m_step=3000,
    # N_iters=10*(10**4), 
    N_iters=3000,
    N_rand=4096,
    ray_sampler='flatten',
    weight_distortion=weight_distortion,
    # pg_scale=[1000, 2000, 3000, 4000, 5000, 6000, 7000],  # default
    pg_scale=[2000, 4000, 6000, 7000],  
    # pg_scale=[],  # used for model size testing
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-6,
    weight_tv_k0=1e-7,
    # added
    pervoxel_lr=False,
    lrate_decay=20,               # default
    lrate_density=1e-1,           # default lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # default lr of the mlp to preduct view-dependent color
    weight_entropy_last=1e-3,     # default
    weight_rgbper=1e-2,           # default
    weight_nearclip=0,
    weight_main=3.0,              # default = 1
    weight_freq=1.0,       
)

coarse_model_and_render = dict(
    maskout_near_cam_vox = maskout_near_cam_vox,
    bbox_thres=1e-10,  # display all the bboxes
)

voxel_size_density = 300  # default 400
voxel_size_rgb = 300  # default 320
voxel_size_viewdir = -1

# voxel_size = 320 # default
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
    contracted_norm='l2',
    rgbnet_dim=3, # default
    rgbnet_direct=True,
    density_type='DenseGrid',
    k0_type='DenseGrid',
    # bg_len=0.2,  # default
    bg_len=0.25,  # default
    viewbase_pe=8,
    maskout_near_cam_vox=False,
)

vis = dict(
    height_rate = 0.6 # camera direction frustrum height
)
