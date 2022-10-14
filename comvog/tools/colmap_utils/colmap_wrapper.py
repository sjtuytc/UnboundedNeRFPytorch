# Source: https://github.com/Fyusion/LLFF
import os
import subprocess
import pdb


# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense
def run_colmap(basedir, match_type):
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    feature_extractor_args = [
        'colmap', 'feature_extractor', 
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--image_path', os.path.join(basedir, 'source'),
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', 'false'
    ]
    joined_args = ' '.join(feature_extractor_args)
    os.system(joined_args)
    # feat_output = ( subprocess.check_output(feature_extractor_args, universal_newlines=True) )
    # logfile.write(feat_output)
    print('Features extracted, the next step would cost several hours, don\'t worry and get a coffee.')

    exhaustive_matcher_args = [
        'colmap', match_type, 
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--SiftMatching.use_gpu', 'false' # comment this line to use gpus, but a desktop is required
    ]
    joined_args = ' '.join(exhaustive_matcher_args)
    print("Executing: ", joined_args)
    os.system(joined_args)
    # match_output = ( subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
    # logfile.write(match_output)
    print('Features matched')
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    mapper_args = [
        'colmap', 'mapper',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, 'source'),
            '--export_path', os.path.join(basedir, 'sparse'), # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
    ]
    joined_args = ' '.join(mapper_args)
    os.system(joined_args)
    # map_output = ( subprocess.check_output(mapper_args, universal_newlines=True) )
    # logfile.write(map_output)
    print('Sparse map created')
    undistorter = [
        'colmap', 'image_undistorter',
        '--image_path', os.path.join(basedir, 'source'),
        '--input_path', os.path.join(basedir, 'sparse', '0'),
        '--output_path', os.path.join(basedir, 'dense'),
        '--output_type', 'COLMAP',
    ]
    joined_args = ' '.join(undistorter)
    os.system(joined_args)
    # undistort_output = subprocess.check_output(undistorter, universal_newlines=True)
    # logfile.write(undistort_output)
    print('Undistort images')
    print( 'Finished running COLMAP! Congrats!')

    # logfile.close()
    # print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )


