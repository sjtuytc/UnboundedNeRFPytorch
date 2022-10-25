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
            '_delete_': True,                           # to ignore the base config
            0   : 1e-4,                                 # 0.5e-5
        }
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
    maskout_near_cam_vox = False
    pervoxel_lr = False
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
    training_ids=[1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,\
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 41, 42, \
            43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, \
                62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, \
                    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,\
                       115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 133, \
                                134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265,\
                                    279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289]
)

coarse_train = dict(
    N_iters=coarse_iter,
    pervoxel_lr = pervoxel_lr,
)

fine_train = dict(
    # N_iters=40000,
    N_iters=100000,
    N_rand=2048,  # reduce this to fit into memory
    # N_rand=4096,  # default
    ray_sampler='flatten',
    weight_distortion=weight_distortion,
    pg_scale=[1000,2000,3000,4000,5000,6000,7000],
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
    weight_main=3.0,              # default = 1
)

coarse_model_and_render = dict(
    maskout_near_cam_vox = maskout_near_cam_vox,
)

# voxel_size = 400  # remember to recover to 400!!!!!!!!!
voxel_size = 320  # default 320
fine_model_and_render = dict(
    num_voxels=voxel_size**3,
    num_voxels_base=voxel_size**3,
    alpha_init=alpha_init,
    stepsize=stepsize,
    fast_color_thres=fast_color_thres,
    world_bound_scale=1,
    # contracted_norm='l2', # default
    rgbnet_dim=3,
    # rgbnet_dim=12, # default
    rgbnet_depth=3, # default
    viewbase_pe=2, # default=4
    bbox_thres=-1,
    maskout_near_cam_vox=False,
    bg_len=0.2,   # default=0.2
)
