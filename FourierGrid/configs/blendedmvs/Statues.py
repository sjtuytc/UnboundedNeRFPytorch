_base_ = '../default.py'

expname = 'dvgo_Statues'
basedir = './logs/blended_mvs'

data = dict(
    datadir='./data/BlendedMVS/Statues/',
    dataset_type='blendedmvs',
    inverse_y=True,
    white_bkgd=True,
)

