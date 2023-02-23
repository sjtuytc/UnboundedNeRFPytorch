_base_ = '../default.py'
seq_name = 'eggbox'
expname = f'{seq_name}_nov8'
basedir = './logs/linemod'

data = dict(
    datadir='./data/linemod',
    dataset_type='linemod',
    white_bkgd=True,
    seq_name=seq_name,
    width_max=140,
    height_max=140
    # 131, 132
)

# fine_train = dict(
#     ray_sampler='flatten',
# )
