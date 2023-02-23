_base_ = '../default.py'
seq_name = 'glue'
expname = f'{seq_name}_nov8'
basedir = './logs/linemod'

data = dict(
    datadir='./data/linemod',
    dataset_type='linemod',
    white_bkgd=True,
    seq_name=seq_name,
    width_max=150,
    height_max=150
    # 111, 147
)

# fine_train = dict(
#     ray_sampler='flatten',
# )
