_base_ = './default_forward_facing.py'

expname = 'sm03_meeting'

data = dict(
    datadir='./data/sm03_meeting/dense',
    factor=2,
    movie_render_kwargs={
        'scale_r': 0.5,
        'scale_f': 1.0,
        'zrate': 1.0,
        'zdelta': 0.5,
    }
)
