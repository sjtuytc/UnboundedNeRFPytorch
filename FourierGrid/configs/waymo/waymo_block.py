_base_ = './waymo_base.py'
model='FourierGrid'
cam_id = 73
expname = f'oct99_waymo_{cam_id}_tt'
vis = dict(
    height_rate = 0.6 # camera direction frustrum height
)

data = dict(
    datadir='data/sep19_ordered_dataset',
    factor=2,
    # near_clip = 0.0356,
    near_clip = 0.1,
    near = 0.1,
    # far = 1,
    far = 0.01,
    # movie_render_kwargs={
    #     'scale_r': 1.0,
    #     'scale_f': 0.8,
    #     'zrate': 2.0,
    #     'zdelta': 0.5,
    # },
    sample_cam=cam_id,
    test_rotate_angle=8, # rotate angle in testing phase
    # sample_idxs=[1127, 11009, 9805, 9426, 5859, 6315]
    sample_interval=1,
    num_per_block=5,  # run this num in block
)

fine_train = dict(
    # N_iters=600, # for quick validation
    N_iters=40000, # 40k for whole training procedure
    # pg_scale=[1000,2000,3000,4000,5000,6000,7000], # default
    pg_scale=[1000,2000,3000,4000,5000,],
)
