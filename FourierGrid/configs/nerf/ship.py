_base_ = '../default.py'

expname = 'dvgo_ship'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/ship',
    dataset_type='blender',
    white_bkgd=True,
)

