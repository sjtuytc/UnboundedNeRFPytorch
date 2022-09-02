# [Weekly classified NeRF](docs/weekly_nerf.md)
We track weekly NeRF papers and classify them. All previous published NeRF papers have been added to the list. We provide an [English version](docs/weekly_nerf.md) and a [Chinese version](docs/weekly_nerf_cn.md). We welcome [contributions and corrections](docs/contribute_weekly_nerf.md) via PR.

We also provide an [excel version](docs/weekly_nerf_meta_data.xlsx) (the meta data) of all NeRF papers, you can add your own comments or make your own paper analysis tools based on the structured meta data.

# Large-scale Neural Radiance Fields in Pytorch

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## 1. Introduction

This project aims for benchmarking several state-of-the-art large-scale neural fields algorithms, not restricted to the original Block-NeRF algorithms. The title of this repo is BlockNeRFPytorch because it is memorizable and short.

The [Block-NeRF](https://waymo.com/intl/zh-cn/research/block-nerf/) builds the largest neural scene representation to date, capable of rendering an entire neighborhood of San Francisco. The abstract of the Block-NeRF paper is as follows:

> We present Block-NeRF, a variant of Neural Radiance Fields that can represent large-scale environments. Specifically, we demonstrate that when scaling NeRF to render city-scale scenes spanning multiple blocks, it is vital to decompose the scene into individually trained NeRFs. This decomposition decouples rendering time from scene size, enables rendering to scale to arbitrarily large environments, and allows per-block updates of the environment. We adopt several architectural changes to make NeRF robust to data captured over months under different environmental conditions. We add appearance embeddings, learned pose refinement, and controllable exposure to each individual NeRF, and introduce a procedure for aligning appearance between adjacent NeRFs so that they can be seamlessly combined. We build a grid of Block-NeRFs from 2.8 million images to create the largest neural scene representation to date, capable of rendering an entire neighborhood of San Francisco.

The official results of Block-NeRF:

https://user-images.githubusercontent.com/31123348/184521599-1b30dea1-a709-4ddd-9287-5c2073d018bf.mp4

This project is the **non-official** implementation of Block-NeRF. You are expected to get the following results in this repository:

1. **Large-scale NeRF training.** The current results are as follows:

https://user-images.githubusercontent.com/31123348/184644052-0e8b33d9-8678-4c95-afe8-d192b309de72.mp4

2. **SOTA custom scenes.** Reconstruction SOTA NeRFs based on your collected photos. Here is a reconstructed video of my work station:

https://user-images.githubusercontent.com/31123348/184643776-fdc4e74d-f901-4cc5-af16-1d28a8097704.mp4

3. **Google Colab support.** Run trained Block-NeRF on Google Colab with detailed visualizations (unfinished yet):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PkzjTlXmGYhovqy68y57LejGmr4XBGrb?usp=sharing)

The other features of this project would be:

- [x] **PyTorch Implementation.** The official Block-NeRF paper uses tensorflow and requires TPUs. However, this implementation only needs PyTorch.

- [x] **GPU efficient.** We ensure that almost all our experiments can be carried on eight NVIDIA 2080Ti GPUs.

- [x] **Quick download.** We host many datasets on Google drive so that downloading becomes much faster.

- [x] **Uniform data format.** The original Block-NeRF paper requires downloading tons of data from Google Cloud Platform. This repo provide processed data and convenient scripts. We provides a uniform data format that suits many datasets of large-scale neural fields.

- [x] **State-of-the-art performance.** This project produces state-of-the-art rendering quality with better efficiency.

- [ ] **Quick validation.** We provide quick validation tools to evaluate your ideas so that you don't need to train on the full Block-NeRF dataset.

- [x] **Open research.** Along with this project, we aim to developping a strong community working on this. We welcome you to joining us (if you have a Wechat, feel free to add my Wechat ytc407). The contributors of this project are listed at the bottom of this page!

- [x] **Chinese community.** We will host regular Chinese tutorials and provide hands-on videos on general NeRF and building your custom NeRFs in the wild and in the city. Welcome to add my Wechat if you have a Wechat.

Welcome to star and watch this project, thank you very much!

## 2. News
- [2022.8.31] Training Mega-NeRF on the Waymo dataset.
- [2022.8.24] Support the full Mega-NeRF pipeline.
- [2022.8.18] Support all previous papers in weekly classified NeRF.
- [2022.8.17] Support classification in weekly NeRF.
- [2022.8.16] Support evaluation scripts and data format standard. Getting some results.
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
2. Install tensorflow, pytorch and other libs. Our version: tensorflow with CUDA11.7.
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install tensorflow 
   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
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

## 4. Large-scale NeRF on the public datasets

Click the following sub-section titles to expand / collapse steps. 

**Note we provide useful debugging commands in many scripts.** Debug commands require a single GPU card only and may run slower than the standard commands. You can use the standard commands instead for conducting experiments and comparisons. A sample bash file is:

```bash
# arguments
ARGUMENTS HERE  # we provide you sampled arguments with explanations and options here.
# for debugging, uncomment the following line when debugging
# DEBUG COMMAND HERE
# for standard training, comment the following line when debugging
STANDARD TRAINING COMMAND HERE
```

<details>
<summary> 4.1 Download processed data and pre-trained models.</summary>

What you should know before downloading the data:

   (1) You don't need these steps if you only want to get results on your custom data (in other words, you can directly go to [Section 5](#5-build-your-custom-large-scale-nerf)) but we recommand you to run on public datasets first.

   (2) **Disclaimer**: you should ensure that you get the permission for usage from the original data provider. One should first sign the license on the [official waymo webiste](https://waymo.com/research/block-nerf/licensing/) to get the permission of downloading the Waymo data. Other data should be downloaded and used without obeying the original licenses.

   (3) Our processed waymo data is significantly **smaller** than the original version (19.1GB vs. 191GB) because we store the camera poses instead of raw ray directions. Besides, our processed data is more friendly for Pytorch dataloaders. 

You can download and preprocess all of the data and pretrained models via the following commands:
```
bash data_proprocess/download_waymo.sh  // download waymo datasets
bash data_preprocess/download_mega.sh   // download mega datasets from the CMU server. The total size is around 31G.
```

(Optional) you may also download the mega dataset (which is the same as the "download\_mega.sh" bash) from [our Google drive](https://drive.google.com/drive/folders/1zzvGWhrbx2_XuK_6mBYpkGngHoL9QGMR?usp=sharing). You can download selected data from this table:

| Dataset name | Images & poses | Masks | Pretrained models |
|---|---|---|---|
| Waymo | [waymo\_image\_poses](https://drive.google.com/file/d/1U7wcE5r-kWtUBscljjTn6q18E8E8kJTd/view?usp=sharing) | Not ready | Not ready |
| Building | [building-pixsfm](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm.tgz) | [building-pixsfm-grid-8](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm-grid-8.tgz) | [building-pixsfm-8.pt](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm-8.pt) |
| Rubble | [rubble-pixsfm](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz) | [rubble-pixsfm-grid-8](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm-grid-8.tgz) | [rubble-pixsfm-8.pt](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm-8.pt) |
| Quad | [ArtsQuad_dataset](http://vision.soic.indiana.edu/disco_files/ArtsQuad_dataset.tar) - [quad-pixsfm](https://storage.cmusatyalab.org/mega-nerf-data/quad-pixsfm.tgz) | [quad-pixsfm-grid-8](https://storage.cmusatyalab.org/mega-nerf-data/quad-pixsfm-grid-8.tgz) | [quad-pixsfm-8.pt](https://storage.cmusatyalab.org/mega-nerf-data/quad-pixsfm-8.pt) |
| Residence | [UrbanScene3D](https://vcc.tech/UrbanScene3D/) - [residence-pixsfm](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm.tgz) | [residence-pixsfm-grid-8](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm-grid-8.tgz) | [residence-pixsfm-8.pt](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm-8.pt) |
| Sci-Art | [UrbanScene3D](https://vcc.tech/UrbanScene3D/) - [sci-art-pixsfm](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm.tgz) | [sci-art-pixsfm-grid-25](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm-grid-25.tgz) | [sci-art-pixsfm-25-w-512.pt](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm-25-w-512.pt) |
| Campus | [UrbanScene3D](https://vcc.tech/UrbanScene3D/) - [campus](https://storage.cmusatyalab.org/mega-nerf-data/campus-pixsfm.tgz) | [campus-pixsfm-grid-8](https://storage.cmusatyalab.org/mega-nerf-data/campus-pixsfm-grid-8.tgz) | [campus-pixsfm-8.pt](https://storage.cmusatyalab.org/mega-nerf-data/campus-pixsfm-8.pt) |

The data structures follow the Mega-NeRF standards. We provide detailed explanations with examples for each data structure in [this doc](docs/mega_format_explained.md). After downloading the data, unzip the files and make folders via the following commands:

```bash
bash data_preprocess/process_mega.sh
```

If you are interested in processing the raw waymo data on your own, please refer to [this doc](./docs/get_pytorch_block_nerf.md).
</details>

<details>
<summary> 4.2 Run pretrained models.</summary>

We recommand you to eval the pretrained models first before you train the models. In this way, you can quickly see the results of our provided models and help you rule out many environmental issues. Run the following script to eval the pre-trained models. The pre-trained models should be downloaded from the previous section 4.1.

```bash
bash scripts/eval_trained_models.sh
# The rendered images would be placed under ${EXP_FOLDER}, which is set to data/mega/${DATASET_NAME}/exp_logs by default.
```
The sample output log by running this script can be found at [docs/sample_logs/eval_trained_models.txt](docs/sample_logs/eval_trained_models.txt).

</details>

<details>
<summary> 4.3 Generate masks.</summary>

Why should we generate masks? (1) Masks help us transfer camera poses + images to ray-based data. In this way, we can download the raw datasets quickly and train quickly as well. (2) Masks helps us manage the boundary of rays.

Run the following script (choose one of the following two cmmands) to create masks:

```bash
bash scripts/create_cluster_mask.sh                      # for the mega dataset
bash scripts/waymo_create_cluster_mask.sh                # for the waymo dataset
# The output would be placed under the ${MASK_PATH}, which is set to data/mega/${DATASET_NAME}/building-pixsfm-grid-8 by default.
```
The sample output log by running this script can be found at [docs/sample_logs/create_cluster_mask.txt](docs/sample_logs/create_cluster_mask.txt). The middle parts of the log have been deleted to save space.
</details>

<details>
<summary> 4.4 Train sub-modules.</summary>

Run the following commands to train the sub-module (the block):
```bash
bash scripts/train_sub_modules.sh SUBMODULE_INDEX         # for the mega dataset
bash scripts/waymo_train_sub_modules.sh SUBMODULE_INDEX   # for the waymo dataset
# SUBMODULE_INDEX is the index of the submodule.
```
The sample output log by running this script can be found at [docs/sample_logs/create_cluster_mask.txt](docs/sample_logs/train_sub_modules.txt). You can also train multiple modules simutaneously via the [parscript](https://github.com/mtli/parscript) to launch all the training procedures simutaneuously. I personally don't use parscript but use the slurm launching scripts to launch all the required modules. The training time without multi-processing is around one day.
</details>

<details>
<summary> 4.5 Merge modules.</summary>

Run the following commands to merge the trained modules to a unified model:
```bash
bash scripts/merge_sub_modules.sh
```
After that, you can go to 4.1 to eval your trained modules. The sample log can be found at [docs/sample_logs/merge_sub_modules.txt](docs/sample_logs/merge_sub_modules.txt).
</details>

## 5. Build your custom large-scale NeRF

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

You may cite this repo to better convince the reviewers about the reproducibility of your paper. If this repo helps you, please cite it as:
```
@software{Zhao_PytorchBlockNeRF_2022,
author = {Zhao, Zelin and Jia, Jiaya},
month = {8},
title = {{PytorchBlockNeRF}},
url = {https://github.com/dvlab-research/BlockNeRFPytorch},
version = {0.0.1},
year = {2022}
}
```

The original paper Block-NeRF and Mega-NeRF can be cited as:

```
 @InProceedings{Tancik_2022_CVPR,
    author    = {Tancik, Matthew and Casser, Vincent and Yan, Xinchen and Pradhan, Sabeek and Mildenhall, Ben and Srinivasan, Pratul P. and Barron, Jonathan T. and Kretzschmar, Henrik},
    title     = {Block-NeRF: Scalable Large Scene Neural View Synthesis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {8248-8258}
}

@inproceedings{turki2022mega,
  title={Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs},
  author={Turki, Haithem and Ramanan, Deva and Satyanarayanan, Mahadev},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12922--12931},
  year={2022}
}
```

We refer to the code and data from [DVGO](https://github.com/sunset1995/DirectVoxGO), [Mega-NeRF](https://github.com/cmusatyalab/mega-nerf), and [SVOX2](https://github.com/sxyu/svox2), thanks for their great work!
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://sjtuytc.github.io/"><img src="https://avatars.githubusercontent.com/u/31123348?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Zelin Zhao</b></sub></a><br /><a href="https://github.com/dvlab-research/BlockNeRFPytorch/commits?author=sjtuytc" title="Code">💻</a> <a href="#maintenance-sjtuytc" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/SEUleaderYang"><img src="https://avatars.githubusercontent.com/u/55042050?v=4?s=100" width="100px;" alt=""/><br /><sub><b>EZ-Yang</b></sub></a><br /><a href="https://github.com/dvlab-research/BlockNeRFPytorch/commits?author=SEUleaderYang" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Alex-Alison-Zhang"><img src="https://avatars.githubusercontent.com/u/71915735?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alex-Zhang</b></sub></a><br /><a href="https://github.com/dvlab-research/BlockNeRFPytorch/issues?q=author%3AAlex-Alison-Zhang" title="Bug reports">🐛</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
