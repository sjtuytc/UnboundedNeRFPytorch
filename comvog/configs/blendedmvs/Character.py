_base_ = '../default.py'

expname = 'dvgo_Character'
basedir = './logs/blended_mvs'

data = dict(
    datadir='./data/BlendedMVS/Character/',
    dataset_type='blendedmvs',
    inverse_y=True,
    white_bkgd=True,
)

