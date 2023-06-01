_base_ = './llff_default.py'
model = 'DVGO'
expname = 'leaves'
basedir = './logs/llff'

data = dict(
    datadir='./data/nerf_llff_data/leaves',
)
