export CONFIG=FourierGrid/configs/waymo/waymo_no_block.py
# python run_FourierGrid.py --program sfm --config ${CONFIG} --sample_num 100 --exp_id 7
python run_FourierGrid.py --program tune_pose --config FourierGrid/configs/waymo/waymo_no_block.py --sample_num 300 --render_video --exp_id 45
# on the mega building dataset
python run_FourierGrid.py --program tune_pose --config FourierGrid/configs/mega/building_no_block.py --sample_num 300 --render_video --exp_id 45
