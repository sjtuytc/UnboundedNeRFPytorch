_base_ = '../default.py'
expname = 'truck_may28_'
vis = dict(
    height_rate = 0.6 # camera direction frustrum height
)
model='FourierGrid'
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
    # fast_color_thres={
    #         '_delete_': True,                           # to ignore the base config
    #         0   : 1e-4,                                 # 0.5e-5
    #     }
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
    weight_distortion = 0.01

data = dict(
    dataset_type='nerfpp',
    inverse_y=True,
    white_bkgd=True,
    rand_bkgd=True,
    unbounded_inward=unbounded_inward,
    load2gpu_on_the_fly=True,
    datadir='./data/tanks_and_temples/tat_training_Truck',
    unbounded_inner_r=1.0,
    ndc=False,
)

coarse_train = dict(
    N_iters=coarse_iter, 
    pervoxel_lr = pervoxel_lr,
)

fine_train = dict(
    N_iters=30000,
    # N_rand=2048,  # reduce this to fit into memory
    N_rand=4096,  # default
    ray_sampler='flatten',
    # ray_sampler='random',
    weight_distortion=weight_distortion,
    pg_scale=[1000, 2000, 3000, 4000, 5000, 6000, 7000],
    # pg_scale=[],
    tv_before=1e9,  # always use tv
    tv_dense_before=10000,
    tv_after=0, # start from beginning
    tv_every=1,
    weight_tv_density=1e-6,
    weight_tv_k0=1e-7,
    pervoxel_lr=False,
    lrate_decay=20,               # default
    lrate_density=1e-1,           # default lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # default lr of the mlp to preduct view-dependent color
    weight_entropy_last=1e-3,     # default
    weight_rgbper=1e-2,           # default
    weight_nearclip=0,
    weight_main=1.0,              # default = 1
    weight_freq=0.0,            
)

coarse_model_and_render = dict(
    maskout_near_cam_vox = maskout_near_cam_vox,
)

voxel_size_density = 200  # default 400
voxel_size_rgb = 200  # default 320
voxel_size_viewdir = -1
# voxel_size_viewdir = 64

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
    # contracted_norm='l2', # default
    rgbnet_dim=12, # default
    fourier_freq_num=4,
    rgbnet_depth=3, # default
    # viewbase_pe=4, # default=4
    bbox_thres=0.001, # should not matter
    maskout_near_cam_vox=False,
    # bg_len=0.2,   # default=0.2
)
