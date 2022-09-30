# render testing sequences
python run_yono.py --program render --config yono/configs/waymo/waymo_tank.py --sample_num 100 --render_test --exp_id 87
# render training sequences
python run_yono.py --program render --config yono/configs/waymo/waymo_tank.py --sample_num 5 --render_train --exp_id 73
python run_yono.py --program render --config yono/configs/tankstemple_unbounded/Playground.py --render_test --render_only --render_video_factor 1
