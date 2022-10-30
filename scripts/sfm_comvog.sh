export CONFIG=comvog/configs/waymo/waymo_no_block.py
python run_comvog.py --program sfm --config ${CONFIG} --sample_num 100 --exp_id 7
