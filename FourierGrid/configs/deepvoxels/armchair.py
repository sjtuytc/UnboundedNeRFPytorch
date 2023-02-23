_base_ = '../default.py'

expname = 'dvgo_armchair'
basedir = './logs/deepvoxels'

data = dict(
    datadir='./data/deepvoxels/',
    dataset_type='deepvoxels',
    scene='armchair',
    white_bkgd=True,
)

