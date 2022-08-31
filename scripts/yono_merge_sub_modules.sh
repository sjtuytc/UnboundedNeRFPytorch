# arguments
export DATASET_NAME=building
export MERGED_OUTPUT=./data/mega/${DATASET_NAME}/${DATASET_NAME}-yono.pt # merge trained models and put to this path
export EXP_FOLDER=data/${DATASET_NAME}/train_exp_logs  # output would be put at this folder
export MASK_PATH=data/mega/${DATASET_NAME}/yono-pixsfm-grid  # output would be put at this folder
export DATASET_PATH=./data/mega/${DATASET_NAME}/${DATASET_NAME}-pixsfm  # raw image folder with poses
# for debugging and standard running
python merge_submodules.py --config_file configs/mega-nerf/${DATASET_NAME}.yaml --ckpt_prefix ${EXP_FOLDER}/ --centroid_path ${MASK_PATH}/params.pt --output $MERGED_OUTPUT