_base_ = '../default.py'

expname = 'dvgo_Steamtrain'
basedir = './logs/nsvf_synthetic'

data = dict(
    datadir='./data/Synthetic_NSVF/Steamtrain',
    dataset_type='nsvf',
    inverse_y=True,
    white_bkgd=True,
)

