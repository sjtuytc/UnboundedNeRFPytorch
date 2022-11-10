_base_ = '../default.py'
seq_name = 'iron'
expname = f'{seq_name}_nov8'
basedir = './logs/linemod'

data = dict(
    datadir='./data/linemod',
    dataset_type='linemod',
    white_bkgd=True,
    seq_name=seq_name,
    width_max=240,
    height_max=240
    # 233, 224
)

# fine_train = dict(
#     ray_sampler='flatten',
# )
