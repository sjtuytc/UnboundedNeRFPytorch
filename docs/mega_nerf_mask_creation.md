# Create Mask when using Mega-NeRF dataset.

Why should we generate masks? (1) Masks help us transfer camera poses + images to ray-based data. In this way, we can download the raw datasets quickly and train quickly as well. (2) Masks helps us manage the boundary of rays.

Run the following script (choose one of the following two cmmands) to create masks:

```bash
bash scripts/create_cluster_mask.sh                      # for the mega dataset
bash scripts/waymo_create_cluster_mask.sh                # for the waymo dataset. Don't be confused here. This command still uses Mega-NeRF algorithm.
# The output would be placed under the ${MASK_PATH}, which is set to data/mega/${DATASET_NAME}/building-pixsfm-grid-8 by default.
```
The sample output log by running this script can be found at [docs/sample_logs/create_cluster_mask.txt](./sample_logs/create_cluster_mask.txt). The middle parts of the log have been deleted to save space.

