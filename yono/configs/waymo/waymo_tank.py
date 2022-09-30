_base_ = './waymo_base.py'
cam_id = 73
expname = f'sep30_waymo_{cam_id}_tt'
vis = dict(
    height_rate = 0.6 # camera direction frustrum height
)
data = dict(
    datadir='data/sep19_ordered_dataset',
    factor=2,
    # near_clip = 0.0356,
    near_clip = 0.01,
    near = 0,
    # far = 1,
    far = 0.01,
    # movie_render_kwargs={
    #     'scale_r': 1.0,
    #     'scale_f': 0.8,
    #     'zrate': 2.0,
    #     'zdelta': 0.5,
    # },
    sample_cam=cam_id,
    # sample_idxs=[1127, 11009, 9805, 9426, 5859, 6315]
)

fine_train = dict(
    N_iters=40000, # 30k is for quick validation
)
