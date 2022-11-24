_base_ = '../default.py'

expname = 'dvgo_Toad'
basedir = './logs/nsvf_synthetic'

data = dict(
    datadir='./data/Synthetic_NSVF/Toad',
    dataset_type='nsvf',
    inverse_y=True,
    white_bkgd=True,
)

