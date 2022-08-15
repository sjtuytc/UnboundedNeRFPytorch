The format of our processed data is:
```bash
pytorch_block_nerf_dataset
    |——————images          // storing all images.
    |        └——————1158877890.png
    |        └——————1480726106.png    
    |        └——————2133100402.png
    |——————json          // storing camera poses and other information
    |        └——————c2w_poses.json // a dict of camera poses of all images
    |        └——————train.json  // a dict of image_name, cam_idx, intrinsics, ....
```