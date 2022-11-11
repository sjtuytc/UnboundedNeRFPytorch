_base_ = '../default.py'
seq_name = 'ape'
expname = f'{seq_name}_nov11_'
basedir = './logs/linemod'

data = dict(
    datadir='./data/linemod',
    dataset_type='linemod',
    white_bkgd=True,
    seq_name=seq_name,
    width_max=90,
    height_max=90
)

fine_train = dict(
    ray_sampler='flatten',
)

voxel_num=32**3
coarse_model_and_render = dict(
    num_voxels_rgb=voxel_num,
    num_voxels_base_rgb=voxel_num,
    num_voxels_density=voxel_num,
    num_voxels_base_density=voxel_num,
)
fine_model_and_render = dict(
    num_voxels_rgb=voxel_num,
    num_voxels_base_rgb=voxel_num,
    num_voxels_density=voxel_num,
    num_voxels_base_density=voxel_num,
    # rgbnet_dim=0,
)