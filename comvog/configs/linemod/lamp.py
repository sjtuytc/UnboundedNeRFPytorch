_base_ = '../default.py'
seq_name = 'lamp'
expname = f'{seq_name}_nov8'
basedir = './logs/linemod'

data = dict(
    datadir='./data/linemod',
    dataset_type='linemod',
    white_bkgd=True,
    seq_name=seq_name,
    width_max=260,
    height_max=260
    # 232, 250
)

# fine_train = dict(
#     ray_sampler='flatten',
# )
