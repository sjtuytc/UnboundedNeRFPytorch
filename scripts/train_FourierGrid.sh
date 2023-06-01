# Free camera dataset
python run_FourierGrid.py --program train --config FourierGrid/configs/free_dataset/grass.py --num_per_block -1 --render_train --render_test --render_video --eval_ssim --eval_lpips_vgg --exp_id 7


# Unbounded tanks and temples
python run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/playground_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 57
python run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/train_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 13
python run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/truck_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 9
python run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/m60_single.py --num_per_block -1 --render_train --render_test --render_video --eval_ssim --eval_lpips_vgg --exp_id 8

# 360 degree dataset
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/room_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 9
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/stump_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 1
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/bicycle_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 12
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/bonsai_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 1
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/garden_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 6
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/kitchen_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/counter_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 9

# Bounded scenes
python run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple/Family_lg.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 11
python run_FourierGrid.py --program train --config FourierGrid/configs/llff/leaves.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 12
python run_FourierGrid.py --program train --config FourierGrid/configs/llff/horns.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 4
python FourierGrid/run_colmap2standard.py --data_dir ./data/nerf_llff_data/horns/
python FourierGrid/run_colmap2standard.py --data_dir ./data/free_dataset/grass/

# nerf-studio dataset
python run_FourierGrid.py  --program train --config FourierGrid/configs/nerf_studio/Giannini_Hall.py --num_per_block -1 --eval_ssim --render_train --render_test --render_video --exp_id 1

# original DVGOv2 training
# python run_FourierGrid.py --program train --config FourierGrid/configs/waymo/block_0_tt.py
# -----------------------------------------------------------------------
# building
python run_FourierGrid.py --program train --config FourierGrid/configs/mega/building_no_block.py --sample_num 300 --render_train --render_video --exp_id 50
python run_FourierGrid.py --program render --config FourierGrid/configs/mega/building.py --sample_num 100 --render_train --exp_id 12 --num_per_block 5  # for render
# rubble
python run_FourierGrid.py --program train --config FourierGrid/configs/mega/rubble.py --sample_num 100 --render_train --num_per_block 5 --exp_id 4  # for train
python run_FourierGrid.py --program train --config FourierGrid/configs/mega/rubble.py --sample_num 10 --render_train --num_per_block 5 --exp_id 4  # for debug
python run_FourierGrid.py --program render --config FourierGrid/configs/mega/rubble.py --sample_num 100 --render_train --num_per_block 5 --exp_id 4  # for render
# quad
python run_FourierGrid.py --program train --config FourierGrid/configs/mega/quad.py --sample_num 100 --render_train --num_per_block 5 --exp_id 1
# lego
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf/lego.py --render_test --eval_ssim --eval_lpips_vgg --exp_id 8
