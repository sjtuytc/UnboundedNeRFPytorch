# for debugging
python create_cluster_masks.py --config configs/mega-nerf/building.yaml --dataset_path data/mega/building/building-pixsfm --output data/mega/building/building-pixsfm-grid-8-debug --grid_dim 2 4
# official
# python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node 8 --max_restarts 0 scripts/create_cluster_masks.py --config configs/mega-nerf/building.yaml --dataset_path ./data/building/building-pixsfm --output ./data/building/building-pixsfm-mask --grid_dim 2 4
