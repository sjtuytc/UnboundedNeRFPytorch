_base_ = './waymo_base.py'

expname = 'sep20_waymo_full'

data = dict(
    datadir='data/sep19_ordered_dataset',
    factor=2,
    movie_render_kwargs={ # not tuned well
        'scale_r': 1.0,
        'scale_f': 0.8,
        'zrate': 2.0,
        'zdelta': 0.5,
    }
)

fine_train = dict(
    N_iters=30000, # 30k is for quick validation
)
