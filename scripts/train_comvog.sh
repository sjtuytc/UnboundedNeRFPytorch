# debugging only five images
python run_comvog.py --program train --config comvog/configs/waymo/waymo_block.py --sample_num 5 --render_test --exp_id 86
python run_comvog.py --program train --config comvog/configs/waymo/waymo_block.py --sample_num 100 --render_train --exp_id 125
python run_comvog.py --program train --config comvog/configs/waymo/waymo_block.py --sample_num 5 --render_train --render_test --exp_id 88
# tanks and temples
python run_comvog.py --program train --config comvog/configs/tankstemple_unbounded/Playground.py --render_train --exp_id 0
# original DVGOv2 training
# python run_comvog.py --program train --config comvog/configs/waymo/block_0_tt.py
# -----------------------------------------------------------------------
# building
python run_comvog.py --program train --config comvog/configs/mega/building.py --sample_num 100 --render_train --exp_id 12  # for train
python run_comvog.py --program train --config comvog/configs/mega/building.py --sample_num 10 --render_train --num_per_block 5 --exp_id 16  # using rgbnet
python run_comvog.py --program train --config comvog/configs/mega/building.py --sample_num 10 --render_train --num_per_block 10 --exp_id 15  # for compare to
python run_comvog.py --program render --config comvog/configs/mega/building.py --sample_num 100 --render_train --exp_id 12 --num_per_block 5  # for render
# rubble
python run_comvog.py --program train --config comvog/configs/mega/rubble.py --sample_num 100 --render_train --num_per_block 5 --exp_id 4  # for train
python run_comvog.py --program train --config comvog/configs/mega/rubble.py --sample_num 10 --render_train --num_per_block 5 --exp_id 4  # for debug
python run_comvog.py --program render --config comvog/configs/mega/rubble.py --sample_num 100 --render_train --num_per_block 5 --exp_id 4  # for render
# quad
python run_comvog.py --program train --config comvog/configs/mega/quad.py --sample_num 100 --render_train --num_per_block 5 --exp_id 1
