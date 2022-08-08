_base_ = '../default.py'

basedir = './logs/custom'

data = dict(
    dataset_type='llff',
    load2gpu_on_the_fly=True,
    ndc=True,
    llffhold=0,
    rand_bkgd=True,
    movie_render_kwargs={
        'scale_r': 1.0, # circling radius
        'scale_f': 1.0, # the distance to the looking point of foucs
        'zdelta': 0.5,  # amplitude of forward motion
        'zrate': 1.0,   # frequency of forward motion
        'N_rots': 1,    # number of rotation in 120 frames
    }
)

coarse_train = dict(
    N_iters=0,
)

fine_train = dict(
    N_iters=30000,
    N_rand=4096,
    weight_distortion=0.01,
    pg_scale=[2000,4000,6000,8000],
    decay_after_scale=0.1,
    ray_sampler='flatten',
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_tv_k0=1e-6,
)

_mpi_depth = 256
_stepsize = 1.0

fine_model_and_render = dict(
    num_voxels=384*384*_mpi_depth,
    mpi_depth=_mpi_depth,
    stepsize=_stepsize,
    rgbnet_dim=9,
    rgbnet_width=64,
    world_bound_scale=1,
    fast_color_thres=_stepsize/_mpi_depth/5,
)