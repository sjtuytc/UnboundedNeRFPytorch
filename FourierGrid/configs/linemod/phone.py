_base_ = '../default.py'
seq_name = 'phone'
expname = f'{seq_name}_nov8'
basedir = './logs/linemod'

data = dict(
    datadir='./data/linemod',
    dataset_type='linemod',
    white_bkgd=True,
    seq_name=seq_name,
    width_max=190,
    height_max=190
    # 159, 187
)

# fine_train = dict(
#     ray_sampler='flatten',
# )
