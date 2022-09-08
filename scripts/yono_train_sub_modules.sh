#!/bin/bash
# arguments
export DATASET_NAME=building
export EXP_FOLDER=data/${DATASET_NAME}/train_exp_logs  # output would be put at this folder
export MASK_PATH=data/mega/${DATASET_NAME}/yono-pixsfm-grid  # output would be put at this folder
export SUBMODULE_INDEX=$1 # submodule index
export DATASET_PATH=./data/mega/${DATASET_NAME}/${DATASET_NAME}-pixsfm  # raw image folder with poses
export NUM_GPUS=4 # number of GPUs
# for debugging
# python train_mega_nerf.py --config_file configs/mega-nerf/${DATASET_NAME}.yaml --exp_name ${EXP_FOLDER}/${SUBMODULE_INDEX} --dataset_path $DATASET_PATH --chunk_paths data/mega/${DATASET_NAME}/chunks/${SUBMODULE_INDEX} --cluster_mask_path ${MASK_PATH}/${SUBMODULE_INDEX} --train_scale_factor 1 --val_scale_factor 1
# for a standard run, comment the following line when debugging
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS train_mega_nerf.py --config_file configs/mega-nerf/${DATASET_NAME}.yaml --exp_name ${EXP_FOLDER}/${SUBMODULE_INDEX} --dataset_path $DATASET_PATH --chunk_paths data/mega/${DATASET_NAME}/chunks/${SUBMODULE_INDEX} --cluster_mask_path ${MASK_PATH}/${SUBMODULE_INDEX} --train_scale_factor 1 --val_scale_factor 1