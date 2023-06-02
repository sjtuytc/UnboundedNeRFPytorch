_base_ = '../default.py'
expname = 'grass_may31_'
vis = dict(
    height_rate = 0.6 # camera direction frustrum height
)
model='FourierGrid'
basedir = './logs/free_dataset'
data = dict(
    datadir='./data/free_dataset/grass',
    dataset_type='free',
    ndc=False,
    training_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, \
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, \
            46, 47, 48, 49, 50],
    pose_scale=0.75,      # scale of pose
    factor=1,
    # inverse_y=True,
    # load2gpu_on_the_fly=True,
    # white_bkgd=True,
    # rand_bkgd=True,
    # movie_render_kwargs={'pitch_deg': 20},
)

coarse_train = dict(
    N_iters = 0,
    pervoxel_lr_downrate=2,
    pervoxel_lr=True,  # DVGO default is True
)

fine_train = dict(
    N_iters=100000,
    N_rand=4096,
    weight_distortion=0.0,
    pg_scale=[2000,4000,6000,8000],
    ray_sampler='flatten',
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_tv_k0=1e-6,
    )

voxel_size_density = 250  # default 400
voxel_size_rgb = 250  # default 320

fine_model_and_render = dict(    
    num_voxels=256**3,
    num_voxels_density=voxel_size_density**3,
    num_voxels_base_density=voxel_size_density**3,
    num_voxels_rgb=voxel_size_rgb**3,
    num_voxels_base_rgb=voxel_size_rgb**3,
    mpi_depth=128,
    rgbnet_dim=9,
    rgbnet_width=64,
    world_bound_scale=1,
    fast_color_thres=1e-3,
)
