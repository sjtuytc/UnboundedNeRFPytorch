# export CONFIG=yono/configs/tankstemple_unbounded/Playground.py #yono/configs/waymo/block_0_tt.py
export CONFIG=yono/configs/waymo/waymo_full.py
# visualize cameras
CUDA_VISIBLE_DEVICES=8 python run_yono.py --program export_bbox --config ${CONFIG} --export_bbox_and_cams_only data/waymo_vis/cam.npz --sample_num 100
# visualize geometry
CUDA_VISIBLE_DEVICES=3 python run_yono.py --program export_coarse --config ${CONFIG} --export_coarse_only data/waymo_vis/cam_coarse.npz
# the following commands require a local desktop
python data_preprocess/visualize_cameras.py --data_path data/samples/block_0 --multi_scale
python tools/vis_train.py data/sep13_block0/cam.npz
python tools/vis_volume.py coarse_mic.npz 0.001 --cam data/sep13_block0/cam_coarse.npz
