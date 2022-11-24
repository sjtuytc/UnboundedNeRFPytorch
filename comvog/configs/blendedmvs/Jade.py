_base_ = '../default.py'

expname = 'dvgo_Jade'
basedir = './logs/blended_mvs'

data = dict(
    datadir='./data/BlendedMVS/Jade/',
    dataset_type='blendedmvs',
    inverse_y=True,
    white_bkgd=False,
)

