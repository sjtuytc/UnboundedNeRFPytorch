_base_ = '../default.py'

expname = 'dvgo_ficus'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/ficus',
    dataset_type='blender',
    white_bkgd=True,
)

