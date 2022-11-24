_base_ = './nerf_unbounded_default.py'

expname = 'dvgo_bicycle_unbounded'

data = dict(
    datadir='./data/360_v2/bicycle',
    factor=4, # 1237x822
    movie_render_kwargs=dict(
        shift_x=0.0,  # positive right
        shift_y=0, # negative down
        shift_z=0,
        scale_r=1.0,
        pitch_deg=-10, # negative look downward
    ),
)

