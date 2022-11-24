_base_ = '../default.py'

expname = 'dvgo_Caterpillar_lg'
basedir = './logs/tanks_and_temple'

data = dict(
    datadir='./data/TanksAndTemple/Caterpillar',
    dataset_type='tankstemple',
    inverse_y=True,
    load2gpu_on_the_fly=True,
    white_bkgd=True,
)

coarse_train = dict(
    pervoxel_lr_downrate=2,
)

fine_train = dict(pg_scale=[1000,2000,3000,4000,5000,6000])
fine_model_and_render = dict(num_voxels=256**3)

