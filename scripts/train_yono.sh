# debugging only five images
python run_yono.py --program train --config yono/configs/waymo/block_0_tt.py --sample_num 5
python run_yono.py --program train --config yono/configs/tankstemple_unbounded/Playground.py  # tanks and temples
# full training
# python run_yono.py --program train --config yono/configs/waymo/block_0_tt.py 