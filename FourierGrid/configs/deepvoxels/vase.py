_base_ = '../default.py'

expname = 'dvgo_vase'
basedir = './logs/deepvoxels'

data = dict(
    datadir='./data/deepvoxels/',
    dataset_type='deepvoxels',
    scene='vase',
    white_bkgd=True,
)

