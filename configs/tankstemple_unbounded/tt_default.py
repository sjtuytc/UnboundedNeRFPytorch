_base_ = '../default.py'

basedir = './logs/tanks_and_temple_unbounded'

data = dict(
    dataset_type='nerfpp',
    inverse_y=True,
    white_bkgd=True,
    rand_bkgd=True,
    unbounded_inward=True,
    load2gpu_on_the_fly=True,
)

coarse_train = dict(N_iters=0)

fine_train = dict(
    N_iters=30000,
    N_rand=4096,
    ray_sampler='flatten',
    weight_distortion=0.01,
    pg_scale=[1000,2000,3000,4000,5000,6000,7000],
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-6,
    weight_tv_k0=1e-7,
)

alpha_init = 1e-4
stepsize = 0.5

fine_model_and_render = dict(
    num_voxels=320**3,
    num_voxels_base=320**3,
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
    contracted_norm='l2',
)

