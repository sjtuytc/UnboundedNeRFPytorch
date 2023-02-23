_base_ = '../default.py'

expname = 'dvgo_donut_369_40208_78816'
basedir = './logs/co3d'

data = dict(
    datadir='./data/co3d/',
    dataset_type='co3d',
    annot_path='./data/co3d/donut/frame_annotations.jgz',
    split_path='./data/co3d/donut/set_lists.json',
    sequence_name='369_40208_78816',
    flip_x=True,
    flip_y=True,
    inverse_y=True,
    white_bkgd=False,
)

coarse_train = dict(
    ray_sampler='flatten',
)

