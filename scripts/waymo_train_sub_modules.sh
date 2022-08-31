#!/bin/bash
# arguments
export DATASET_NAME=pytorch_waymo_dataset
export EXP_FOLDER=data/${DATASET_NAME}/train_exp_logs  # output would be put at this folder
export MASK_PATH=data/${DATASET_NAME}/pixsfm-grid-8 # output would be put at this folder
export DATASET_PATH=./data/${DATASET_NAME}  # raw image folder with poses
export NUM_GPUS=1 # number of GPUs
export SUBMODULE_INDEX=$1 # submodule index
# for debugging
python train.py --config_file configs/mega-nerf/${DATASET_NAME}.yaml --exp_name ${EXP_FOLDER}/${SUBMODULE_INDEX} --dataset_path $DATASET_PATH --chunk_paths data/${DATASET_NAME}/chunks/${SUBMODULE_INDEX} --cluster_mask_path ${MASK_PATH}/${SUBMODULE_INDEX} --train_scale_factor 1 --val_scale_factor 1
