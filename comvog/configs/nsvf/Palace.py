_base_ = '../default.py'

expname = 'dvgo_Palace'
basedir = './logs/nsvf_synthetic'

data = dict(
    datadir='./data/Synthetic_NSVF/Palace',
    dataset_type='nsvf',
    inverse_y=True,
    white_bkgd=True,
)

