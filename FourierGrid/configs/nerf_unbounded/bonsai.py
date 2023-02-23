_base_ = './nerf_unbounded_default.py'

expname = 'dvgo_bonsai_unbounded'

data = dict(
    datadir='./data/360_v2/bonsai',
    factor=2, # 1559x1039
    movie_render_kwargs=dict(
        shift_x=0.0,  # positive right
        shift_y=0, # negative down
        shift_z=0,
        scale_r=1.0,
        pitch_deg=-30, # negative look downward
    ),
)

