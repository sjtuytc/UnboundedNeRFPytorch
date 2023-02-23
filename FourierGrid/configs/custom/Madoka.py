_base_ = './default_forward_facing.py'

expname = 'Madoka'

data = dict(
    datadir='data/Madoka/dense',
    factor=2,
    movie_render_kwargs={
        'scale_r': 1.0,
        'scale_f': 0.8,
        'zrate': 2.0,
        'zdelta': 0.5,
    }
)

fine_train = dict(
    N_iters=300000,
)