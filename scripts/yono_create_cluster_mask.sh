# arguments
export DATASET_NAME=building
export MASK_PATH=data/mega/${DATASET_NAME}/yono-pixsfm-grid  # output would be put at this folder
export DATASET_PATH=./data/mega/${DATASET_NAME}/${DATASET_NAME}-pixsfm  # raw image folder with poses
export NUM_GPUS=4 # number of GPUs
# for debugging only, this is slow.
# python create_cluster_masks.py --config configs/mega-nerf/${DATASET_NAME}.yaml --dataset_path ${DATASET_PATH} --output ${MASK_PATH}-debug --grid_dim 1 1
# for a standard run, comment the following line when debugging
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node ${NUM_GPUS} --max_restarts 0 create_cluster_masks.py --config configs/mega-nerf/${DATASET_NAME}.yaml --dataset_path ${DATASET_PATH} --output ${MASK_PATH} --grid_dim 1 1
