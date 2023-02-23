_base_ = '../default.py'
seq_name = 'benchvise'
expname = f'{seq_name}_nov9_'
basedir = './logs/linemod'

data = dict(
    datadir='./data/linemod',
    dataset_type='linemod',
    white_bkgd=True,
    seq_name=seq_name,
    width_max=230,
    height_max=230,
    #198, 224
)

# fine_train = dict(
#     ray_sampler='flatten',
# )
