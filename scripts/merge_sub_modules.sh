# arguments
export DATASET_NAME=building
export EXP_FOLDER=data/mega/${DATASET_NAME}/train_exp_logs/  # load checkpoints from this folder
export MERGED_OUTPUT=./data/mega/${DATASET_NAME}/${DATASET_NAME}-pixsfm-8.pt # merge trained models and put to this path
export MASK_PATH=data/mega/${DATASET_NAME}/building-pixsfm-grid-8  # load mask from this path
# for debugging and standard running
python merge_submodules.py --config_file mega_nerf/configs/${DATASET_NAME}.yaml --ckpt_prefix ${EXP_FOLDER}/ --centroid_path ${MASK_PATH}/params.pt --output $MERGED_OUTPUT