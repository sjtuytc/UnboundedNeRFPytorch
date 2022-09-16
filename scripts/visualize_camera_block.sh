export CONFIG=configs/tankstemple_unbounded/Playground.py #configs/waymo/block_0_tt.py
# visualize cameras
CUDA_VISIBLE_DEVICES=8 python run.py --config ${CONFIG} --export_bbox_and_cams_only data/tanks_vis/cam.npz
# visualize geometry
CUDA_VISIBLE_DEVICES=8 python run.py --config ${CONFIG} --export_coarse_only data/tanks_vis/cam_coarse.npz
# the following commands require a local desktop
python data_preprocess/visualize_cameras.py --data_path data/samples/block_0 --multi_scale
python tools/vis_train.py data/sep13_block0/cam.npz
python tools/vis_volume.py coarse_mic.npz 0.001 --cam data/sep13_block0/cam_coarse.npz
