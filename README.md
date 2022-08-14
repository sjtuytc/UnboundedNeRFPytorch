# We track weekly NeRF papers [here](docs/weekly_nerf.md) and the Chinese version is [here](docs/weekly_nerf_cn.md).
# [CVPR22Oral] Block-NeRF: Scalable Large Scene Neural View Synthesis
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
## 1. Introduction

The [Block-NeRF](https://waymo.com/intl/zh-cn/research/block-nerf/) builds the largest neural scene representation to date, capable of rendering an entire neighborhood of San Francisco. The abstract of the Block-NeRF paper is as follows:

> We present Block-NeRF, a variant of Neural Radiance Fields that can represent large-scale environments. Specifically, we demonstrate that when scaling NeRF to render city-scale scenes spanning multiple blocks, it is vital to decompose the scene into individually trained NeRFs. This decomposition decouples rendering time from scene size, enables rendering to scale to arbitrarily large environments, and allows per-block updates of the environment. We adopt several architectural changes to make NeRF robust to data captured over months under different environmental conditions. We add appearance embeddings, learned pose refinement, and controllable exposure to each individual NeRF, and introduce a procedure for aligning appearance between adjacent NeRFs so that they can be seamlessly combined. We build a grid of Block-NeRFs from 2.8 million images to create the largest neural scene representation to date, capable of rendering an entire neighborhood of San Francisco.

The official results of Block-NeRF:

https://user-images.githubusercontent.com/31123348/184521599-1b30dea1-a709-4ddd-9287-5c2073d018bf.mp4

This project is the **non-official** implementation of Block-NeRF. Ideally, the features of this project would be:

- **PyTorch Implementation.** The official Block-NeRF paper uses tensorflow and requires TPUs. However, this implementation only needs PyTorch.
- **Better data preprocessing.** The original Block-NeRF paper requires downloading tons of data from Google Cloud Platform. This repo provide processed data and convenient scripts.
- **State-of-the-art performance.** This project produces state-of-the-art rendering quality with better efficiency.

- **Quick validation.** We provide quick validation tools to evaluate your ideas so that you don't need to train on the full Block-NeRF dataset.

- **Open research and better community.** Along with this project, we aim to developping a strong community working on this. We welcome you to joining us (if you have a Wechat, feel free to add my Wechat ytc407). The contributors of this project are listed at the bottom of this page!

You are expected to get the following results in this repository:

1. **SOTA custom scenes.** Reconstruction SOTA NeRFs based on your collected photos.

   <img src="figs/sm01_04.gif" alt="drawing" width="200"/>


2. **Block NeRF training.** Training BlockNeRFs. The results will be released here in the next commit. Stay tuned!

Welcome to watch this project!

## 2. News
- [2022.8.13] Add estimated camera pose and release a better dataset.
- [2022.8.12] Add weekly NeRF functions.
- [2022.8.8] Add the NeRF reconstruction code and doc for custom purposes.
- [2022.7.28] The data preprocess script is finished.
- [2022.7.20] This project started!

## 3. Installation
<details>
<summary>Expand / collapse installation steps.</summary>

1. Create conda environment.
   ```bash
   conda create -n nerf-block python=3.9
   ```
2. Install tensorflow and other libs. You don't need to install tensorflow if you download our processed data. Our version: tensorflow with CUDA11.7.
   ```bash
   pip install tensorflow opencv-python matplotlib
   ```
3. Install other libs used for reconstructing custom scenes, which is only needed when you need to build your scenes.
   ```bash
   sudo apt-get install colmap
   sudo apt-get install imagemagick  # required sudo accesss
   pip install -r requirements.txt
   conda install pytorch-scatter -c pyg  # or install via https://github.com/rusty1s/pytorch_scatter
   ```
   You can use laptop version of COLMAP as well if you do not have access to sudo access on your server. However, we found if you do not set up COLMAP parameters properly, you would not get the SOTA performance.
</details>

## 4. Data preprocess, training and eval

<details>
<summary>Expand / collapse steps for the pipeline of Block-NeRF.</summary>
You don't need these steps if you only want to get results on your custom data.

1. One should first sign the license on the [official waymo webiste](https://waymo.com/research/block-nerf/licensing/) to get the permission of downloading the data.

2. You can download our processed data directly at [this link](https://drive.google.com/file/d/1U7wcE5r-kWtUBscljjTn6q18E8E8kJTd/view?usp=sharing). Our processed data is significantly smaller than the original version (19.1GB vs. 191GB) because we store the camera poses instead of raw ray directions. Besides, our processed data is more friendly for Pytorch dataloaders. 
   You can download the processed data via the following commands as well:
   ```bash
   gdown --id 1U7wcE5r-kWtUBscljjTn6q18E8E8kJTd
   unzip pytorch_block_nerf_dataset.zip
   ```
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
   If you are interested in creating the data by your own, please refer to [this page](docs/get_pytorch_block_nerf.md). 

</details>

## 5. Build custom NeRF world

<details>
<summary>Expand / collapse steps for building custom NeRF world.</summary>

1. Put your images under data folder. The structure should be like:

	```bash
	data
	   |——————Madoka          // Your folder name here.
	   |        └——————source // Source images should be put here.
	   |                 └——————---|1.png
	   |                 └——————---|2.png
	   |                 └——————---|...
	```
   The sample data is provided in [our Google drive folder](https://drive.google.com/drive/folders/1JyX0VNf0R58s46Abj8HDO1NwZqmGOVRS?usp=sharing). The Madoka and Otobai can be found [at this link](https://sunset1995.github.io/dvgo/tutor_forward_facing.html). 

2. Run COLMAP to reconstruct scenes. This would probably cost a long time.

	```bash
	python tools/imgs2poses.py data/Madoka
	```
   You can replace data/Madoka by your data folder.
   If your COLMAP version is larger than 3.6 (which should not happen if you use apt-get), you need to change export_path to output_path in Ln67 of colmap_wrapper.py.

3. Training NeRF scenes.

	```bash
	python run.py --config configs/custom/Madoka.py
	```
   You can replace configs/custom/Madoka.py by other configs.
4. Validating the training results to generate a fly-through video.

	```bash
	python run.py --config configs/custom/Madoka.py --render_only --render_video --render_video_factor 8
	```
</details>


## 6. Citations & acknowledgements

If this repo helps you, please cite it as:
```bash
@software{Zhao_PytorchBlockNeRF_2022,
author = {Zhao, Zelin and Jia, Jiaya},
month = {8},
title = {{PytorchBlockNeRF}},
url = {https://github.com/dvlab-research/BlockNeRFPytorch},
version = {0.0.1},
year = {2022}
}
```

You may cite this repo to better convince the reviewers about the reproducibility of your paper.

The original paper Block-NeRF can be cited as:

```bash
 @InProceedings{Tancik_2022_CVPR,
    author    = {Tancik, Matthew and Casser, Vincent and Yan, Xinchen and Pradhan, Sabeek and Mildenhall, Ben and Srinivasan, Pratul P. and Barron, Jonathan T. and Kretzschmar, Henrik},
    title     = {Block-NeRF: Scalable Large Scene Neural View Synthesis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {8248-8258}
}
```

We refer to the code and data from [DVGO](https://github.com/sunset1995/DirectVoxGO) and [SVOX2](https://github.com/sxyu/svox2), thanks for their great work!
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
   <td align="center"><a href="https://sjtuytc.github.io/"><img src="https://avatars.githubusercontent.com/u/31123348?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Zelin Zhao</b></sub></a><br /><a href="https://github.com/dvlab-research/BlockNeRFPytorch/commits?author=sjtuytc" title="Code">💻</a> <a href="#maintenance-sjtuytc" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/SEUleaderYang"><img src="https://avatars.githubusercontent.com/u/55042050?v=4?s=100" width="100px;" alt=""/><br /><sub><b>EZ-Yang</b></sub></a><br /><a href="https://github.com/dvlab-research/BlockNeRFPytorch/commits?author=SEUleaderYang" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
