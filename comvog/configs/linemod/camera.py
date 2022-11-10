_base_ = '../default.py'
seq_name = 'camera'
expname = f'{seq_name}_nov9_'
basedir = './logs/linemod'

data = dict(
    datadir='./data/linemod',
    dataset_type='linemod',
    white_bkgd=True,
    seq_name=seq_name,
    width_max=150,
    height_max=150
    # 142, 137
)

# fine_train = dict(
#     ray_sampler='flatten',
# )
