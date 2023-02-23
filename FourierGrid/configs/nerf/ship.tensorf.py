_base_ = '../default.py'

expname = 'dvgo_ship_tensorf'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/ship',
    dataset_type='blender',
    white_bkgd=True,
)

fine_train = dict(
    lrate_density=0.02,
    lrate_k0=0.02,
    pg_scale=[1000,2000,3000,4000,5000,6000],
)

fine_model_and_render = dict(
    num_voxels=384**3,
    density_type='TensoRFGrid',
    density_config=dict(n_comp=8),
    k0_type='TensoRFGrid',
    k0_config=dict(n_comp=24),
)

