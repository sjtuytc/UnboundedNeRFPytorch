_base_ = '../default.py'
model='FourierGrid'
basedir = 'logs/waymo'
visualize_poses = False
alpha_init = 1e-4
stepsize = 0.5
_mpi_depth = 256
maskout_near_cam_vox = False  # changed
pervoxel_lr = False
unbounded_inward = True
expname = f'oct29_waymo'
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
    dataset_type='waymo',
    inverse_y=True,
    white_bkgd=True,     # almost no effect when rand_bkgd=True
    rand_bkgd=True,      # random background
    unbounded_inward=unbounded_inward,
    load2gpu_on_the_fly=True,
    datadir='data/sep19_ordered_dataset',
    factor=2,
    near_clip = 0.1,
    near = 0.1,
    far = 0.01,
    # sample_cam=cam_id,
    test_rotate_angle=360, # rotate angle in testing phase
    sample_interval=1,
    num_per_block=-1,  # run this num in block
    unbounded_inner_r=0.8,
    # three views
    # training_ids=['69_0', '71_0', '73_0'], 
    training_ids=['73_0', '73_1', '73_2', '73_3', '73_4', '73_5', '73_6', '73_7', '73_8', '73_9', \
        '73_10', '73_11', '73_12', '73_13', '73_14', '73_15', '73_16', '73_17', '73_18', '73_19', \
            '73_20', '73_21', '73_22', '73_23', '73_24', '73_25', '73_26', '73_27', '73_28', '73_29', \
                '73_30', '73_31', '73_32', '73_33', '73_34', '73_35', '73_36', '73_37', '73_38', '73_39', \
                    '73_40', '73_41', '73_42', '73_43', '73_44', '73_45', '73_46', '73_47', '73_48', '73_49'],
    tunning_id = '71_0',
    search_rot_lower = [129, -2, -2],
    search_rot_upper = [133, 2, 2],
    search_pos_lower = [0.0, -0.01, -0.01],
    search_pos_upper = [0.04, 0.01, 0.01],
    search_num = 10**4,
    # assign_pos = {
    #     '69_0': [0.0, 0.0, 0.0], 
    #               '71_0': [0.03251668821656803, 0.001401165785078217, 0.00560227169424881], 
    #               '73_0': [0.0, 0.0, 0.0]
    #               },
    # assign_rot = {
        # '69_0': [175, 0.0, 0.0],
                # '71_0': [132.27749560775322, -1.1274407139317342, -0.42476203263358325],
                # '73_0': [85.2267753, 0.0, 0.0]
                # },
    # assign_pos = {
    #     '69_0': [0.0, 0.0, 0.0], 
    #     '69_10': [0.0, -3.55 * 0.01, 0.0], 
    #     '69_20': [0.0, -4.45 * 0.01, 0.0], 
    #     '69_30': [0.0, -5.84 * 0.01, 0.0], 
    #     '69_40': [0.0, -6.99 * 0.01, 0.0], 
    #     '69_50': [0.0, -8.66 * 0.01, 0.0],
    # },
    # assign_rot = {
    #     '69_0': [175, 0.0, 0.0],
    #     '69_10': [175, 0.0, 0.0],
    #     '69_20': [175, 0.0, 0.0], 
    #     '69_30': [175, 0.0, 0.0], 
    #     '69_40': [175, 0.0, 0.0], 
    #     '69_50': [175, 0.0, 0.0],
    # }
)

coarse_train = dict(
    N_iters=coarse_iter,
    pervoxel_lr = pervoxel_lr,
    ray_sampler='flatten',
)

fine_train = dict(
    N_iters_m_step=1500,            # search via sfm
    N_iters=3000,
    # N_iters=10*(10**4),
    N_rand=2048,
    ray_sampler='flatten',
    weight_distortion=weight_distortion,
    pg_scale=[3000, 4000, 5000, 6000, 7000],
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

diffusion = dict(
    diff_root = 'diffusion',
    diff_replace = {'69_0': 'airplane'}    
)

coarse_model_and_render = dict(
    maskout_near_cam_vox = maskout_near_cam_vox,
    bbox_thres=1e-10,  # display all the bboxes
)

voxel_size_density = 300  # default 400
voxel_size_rgb = 300  # default 320
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
    contracted_norm='l2',
    rgbnet_dim=3, # default
    rgbnet_direct=True,
    density_type='DenseGrid',
    k0_type='DenseGrid',
    bg_len=0.2,  # very important
    viewbase_pe=2,
    maskout_near_cam_vox=False,
)

vis = dict(
    height_rate = 0.6 # camera direction frustrum height
)
