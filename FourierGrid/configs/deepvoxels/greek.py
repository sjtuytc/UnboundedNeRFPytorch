_base_ = '../default.py'

expname = 'dvgo_greek'
basedir = './logs/deepvoxels'

data = dict(
    datadir='./data/deepvoxels/',
    dataset_type='deepvoxels',
    scene='greek',
    white_bkgd=True,
)

