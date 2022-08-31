# arguments
export DATASET_NAME=building
export NUM_GPUS=2 # number of GPUs
export EXP_FOLDER=data/${DATASET_NAME}/train_exp_logs  # output would be put at this folder
export DATASET_PATH=./data/mega/${DATASET_NAME}/${DATASET_NAME}-pixsfm  # raw image folder with poses
export MERGED_OUTPUT=./data/mega/${DATASET_NAME}/${DATASET_NAME}-yono.pt # merge trained models and put to this path
# for debugging, uncomment the following line when debugging
python eval.py --config_file configs/mega-nerf/${DATASET_NAME}.yaml  --exp_name $EXP_FOLDER --dataset_path $DATASET_PATH --container_path $MERGED_OUTPUT
# for a standard run, comment the following line when debugging
# python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS eval.py --config_file configs/mega-nerf/${DATASET_NAME}.yaml  --exp_name $EXP_FOLDER --dataset_path $DATASET_PATH --container_path $MERGED_OUTPUT