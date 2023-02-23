_base_ = '../default.py'
seq_name = 'driller'
expname = f'{seq_name}_nov8'
basedir = './logs/linemod'

data = dict(
    datadir='./data/linemod',
    dataset_type='linemod',
    white_bkgd=True,
    seq_name=seq_name,
    width_max=250,
    height_max=250,
    # 240, 237
)

# fine_train = dict(
#     ray_sampler='flatten',
# )
