# debugging only five images
python run_yono.py --program train --config yono/configs/waymo/waymo_tank.py --sample_num 5 --render_test --exp_id 86
python run_yono.py --program train --config yono/configs/waymo/waymo_tank.py --sample_num 100 --render_test --exp_id 87
python run_yono.py --program train --config yono/configs/waymo/waymo_tank.py --sample_num 5 --render_train --render_test --exp_id 85
# python run_yono.py --program train --config yono/configs/tankstemple_unbounded/Playground.py --render_train # tanks and temples
# full training
# python run_yono.py --program train --config yono/configs/waymo/block_0_tt.py