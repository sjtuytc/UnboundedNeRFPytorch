_base_ = './nerf_unbounded_default.py'

expname = 'dvgo_room_unbounded'

data = dict(
    datadir='./data/360_v2/room',
    factor=2, # 1557x1038
    movie_render_kwargs=dict(
        shift_x=0.0,  # positive right
        shift_y=-0.3, # negative down
        shift_z=0,
        scale_r=0.2,
        pitch_deg=-40, # negative look downward
    ),
)

