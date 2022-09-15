CUDA_VISIBLE_DEVICES=8 python run.py --config configs/waymo/block_0_tt.py --export_bbox_and_cams_only data/sep13_block0/cam.npz
CUDA_VISIBLE_DEVICES=8 python run.py --config configs/waymo/block_0_tt.py --export_coarse_only data/sep13_block0/cam_coarse.npz
# the following commands require a local desktop
python data_preprocess/visualize_cameras.py --data_path data/samples/block_0 --multi_scale
python tools/vis_train.py data/sep13_block0/cam.npz
python tools/vis_volume.py coarse_mic.npz 0.001 --cam data/sep13_block0/cam_coarse.npz
