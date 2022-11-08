# [Weekly classified NeRF](docs/weekly_nerf.md)
We track weekly NeRF papers and classify them. All previous published NeRF papers have been added to the list. We provide an [English version](docs/weekly_nerf.md) and a [Chinese version](docs/weekly_nerf_cn.md). We welcome [contributions and corrections](docs/contribute_weekly_nerf.md) via PR.

We also provide an [excel version](docs/weekly_nerf_meta_data.xlsx) (the meta data) of all NeRF papers, you can add your own comments or make your own paper analysis tools based on the structured meta data.

# Large-scale Neural Radiance Fields in Pytorch

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## 1. Introduction

**Since I changed my research direction, the updates on the following codes might be slow. Many improvements in this repo are not going to be published at this moment, so feel free to use it. The weekly NeRF would be updated as usual.**

This project aims for benchmarking several state-of-the-art large-scale radiance fields algorithms, not restricted to the original Block-NeRF algorithm.

The [Block-NeRF](https://waymo.com/intl/zh-cn/research/block-nerf/) builds the largest neural scene representation to date, capable of rendering an entire neighborhood of San Francisco.

This project is the **non-official** implementation of Block-NeRF. You are expected to get the following results in this repository:

1. **Large-scale NeRF training.** The current results are as follows:

Training splits:

https://user-images.githubusercontent.com/31123348/200509378-4b9fe63f-4fa4-40b1-83a9-b8950d981a3b.mp4

Rotation: 

https://user-images.githubusercontent.com/31123348/200509910-a5d8f820-143a-4e03-8221-b04d0db2d050.mov

2. **SOTA custom scenes.** Reconstruction SOTA NeRFs based on your collected photos. Here is a reconstructed video of my work station:

https://user-images.githubusercontent.com/31123348/184643776-fdc4e74d-f901-4cc5-af16-1d28a8097704.mp4

3. **Google Colab support.** Run trained Block-NeRF on Google Colab with detailed visualizations (unfinished yet):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PkzjTlXmGYhovqy68y57LejGmr4XBGrb?usp=sharing)

The other features of this project would be:

- [x] **PyTorch Implementation.** The official Block-NeRF paper uses tensorflow and requires TPUs. However, this implementation only needs PyTorch.

- [x] **GPU efficient.** We ensure that almost all our experiments can be carried on eight NVIDIA 2080Ti GPUs.

- [x] **Quick download.** We host many datasets on Google drive so that downloading becomes much faster.

- [x] **Uniform data format.** The original Block-NeRF paper requires downloading tons of data from Google Cloud Platform. This repo provide processed data and convenient scripts. We provides a uniform data format that suits many datasets of large-scale neural fields.

- [ ] **State-of-the-art performance.** This project produces state-of-the-art rendering quality with better efficiency.

- [ ] **Quick validation.** We provide quick validation tools to evaluate your ideas so that you don't need to train on the full Block-NeRF dataset.

- [x] **Open research.** Along with this project, we aim to developping a strong community working on this. We welcome you to joining us (if you have a Wechat, feel free to add my Wechat ytc407). The contributors of this project are listed at the bottom of this page.


Hope our efforts could help your research or projects!

## 2. News
- [2022.9.12] Training Block-NeRF on the Waymo dataset, reaching PSNR 24.3.
- [2022.8.31] Training Mega-NeRF on the Waymo dataset, loss still NAN.
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
   conda install pytorch-scatter -c pyg  # or install via https://github.com/rusty1s/pytorch_scatter
   ```
   You can use laptop version of COLMAP as well if you do not have access to sudo access on your server. However, we found if you do not set up COLMAP parameters properly, you would not get the SOTA performance.
</details>

## 4. Large-scale NeRF on the public datasets

**We provide implementations for two algorithms: Block-NeRF and Mega-NeRF.** Most of the Mega-NeRF implementations is from [the official Mega-NeRF repo](https://github.com/cmusatyalab/mega-nerf) while we support [Waymo dataset training](https://waymo.com/intl/zh-cn/research/block-nerf/), visualizations and fix some Mega-NeRF bugs. In the following, the Mega-NeRF commands are commented to prevent confusions.

**We provide useful debugging commands in many scripts.** Debug commands require a single GPU card only and may run slower than the standard commands. You can use the standard commands instead for conducting experiments and comparisons. A sample bash file is:

```bash
# arguments
ARGUMENTS HERE  # we provide you sampled arguments with explanations and options here.
# for debugging, uncomment the following line when debugging
# DEBUG COMMAND HERE
# for standard training, comment the following line when debugging
STANDARD TRAINING COMMAND HERE
```

Click the following sub-section titles to expand / collapse steps.

<details>
<summary> 4.1 Download processed data and pre-trained models.</summary>

What you should know before downloading the data:

   (1) **Disclaimer**: you should ensure that you get the permission for usage from the original data provider. One should first sign the license on the [official waymo webiste](https://waymo.com/research/block-nerf/licensing/) to get the permission of downloading the Waymo data. Other data should be downloaded and used without obeying the original licenses.

   (2) Our processed waymo data is significantly **smaller** than the original version (19.1GB vs. 191GB) because we store the camera poses instead of raw ray directions. Besides, our processed data is more friendly for Pytorch dataloaders. Furthermore, the processed data support training by Mega-NeRF and Block-NeRF both.

Download [the data](https://drive.google.com/drive/folders/1Lcc6MF35EnXGyUy0UZPkUx7SfeLsv8u9?usp=sharing) and [pretrained models](https://drive.google.com/drive/folders/1O7uzcPBQHNAcmAcmcS6TRbLqiIDE3D0y?usp=sharing) in the Google Drive. You may use [gdown](https://stackoverflow.com/questions/65001496/how-to-download-a-google-drive-folder-using-link-in-linux) to download the files via command lines.

If you are interested in processing the raw waymo data on your own, please refer to [this doc](./docs/get_pytorch_waymo_dataset.md).

The downloaded data would look like this:

   ```
   data
      |——————pytorch_waymo_dataset                     // the root folder for pytorch waymo dataset
      |        └——————cam_info.json                    // extracted cam2img information in dict.
      |        └——————coordinates.pt                   // global camera information used in Mega-NeRF
      |        └——————train                            // train data
      |        |         └——————metadata               // meta data per image (camera information, etc)
      |        |         └——————rgbs                   // rgb images
      |        |         └——————split_block_train.json // split block informations
      |        |         └——————train_all_meta.json    // all meta informations in train folder
      |        └——————val                              // val data with the same structure as train
   ```

If you wish to run the Mega-NeRF algorithm, you will need to create masks prior to the training or evaluation. Please refer to [this doc](./docs/mega_nerf_mask_creation.md) for more details. You can download other Mega-NeRF benchmarks following [this doc](./docs/download_and_process_mega.md).

</details>

<details>
<summary> 4.2 Run pretrained models.</summary>

We recommand you to eval the pretrained models first before you train the models. In this way, you can quickly see the results of our provided models and help you rule out many environmental issues. Run the following script to eval the pre-trained models, which should be downloaded from the previous section 4.1.

```bash
bash scripts/block_nerf_eval.sh
# bash scripts/mega_nerf_eval.sh  # for the Mega-NeRF algorithm. The rendered images would be placed under ${EXP_FOLDER}, which is set to data/mega/${DATASET_NAME}/exp_logs by default. The sample output log by running this script can be found at [docs/sample_logs/mega_nerf_eval.txt](docs/sample_logs/mega_nerf_eval.txt).
```

</details>

<details>
<summary> 4.3 Train sub-modules.</summary>

Run the following commands to train the sub-modules (the blocks):
```bash
export BLOCK_INDEX=0
bash scripts/block_nerf_train.sh ${BLOCK_INDEX}                   # For the Block-NeRF algorithm. The training tensorboard log is at the logs/. Using "tensorboard dev --logdir logs/" to see the tensorboard log. 

# bash scripts/mega_nerf_train_sub_modules.sh ${BLOCK_INDEX}      # For the Mega-NeRF algorithm. The sample training log is at[docs/sample_logs/mega_nerf_train_sub_modules.txt](docs/sample_logs/mega_nerf_train_sub_modules.txt) . You can also train multiple modules simutaneously via the [parscript](https://github.com/mtli/parscript) to launch all the training procedures simutaneuously. I personally don't use parscript but use the slurm launching scripts to launch all the required modules. The training time without multi-processing is around one day.

# If you are running the Mega-NeRF algorithm, you need to merge the trained modules:
# ```bash
# bash scripts/merge_sub_modules.sh
# ```
# The sample log can be found at [docs/sample_logs/merge_sub_modules.txt](docs/sample_logs/merge_sub_modules.txt).
```
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
   If your COLMAP version is larger than 3.6 (which should not happen if you use apt-get), you need to change export_path to output_path in the colmap_wrapper.py.

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

We refer to the code and data from [DVGO](https://github.com/sunset1995/DirectVoxGO), [Mega-NeRF](https://github.com/cmusatyalab/mega-nerf), [nerf-pl](https://github.com/kwea123/nerf_pl) and [SVOX2](https://github.com/sxyu/svox2), thanks for their great work!
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://sjtuytc.github.io/"><img src="https://avatars.githubusercontent.com/u/31123348?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Zelin Zhao</b></sub></a><br /><a href="https://github.com/dvlab-research/LargeScaleNeRFPytorch/commits?author=sjtuytc" title="Code">💻</a> <a href="#maintenance-sjtuytc" title="Maintenance">🚧</a></td>
      <td align="center"><a href="https://github.com/SEUleaderYang"><img src="https://avatars.githubusercontent.com/u/55042050?v=4?s=100" width="100px;" alt=""/><br /><sub><b>EZ-Yang</b></sub></a><br /><a href="https://github.com/dvlab-research/LargeScaleNeRFPytorch/commits?author=SEUleaderYang" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/Alex-Alison-Zhang"><img src="https://avatars.githubusercontent.com/u/71915735?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alex-Zhang</b></sub></a><br /><a href="https://github.com/dvlab-research/LargeScaleNeRFPytorch/issues?q=author%3AAlex-Alison-Zhang" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://fanlu97.github.io/"><img src="https://avatars.githubusercontent.com/u/45007531?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Fan Lu</b></sub></a><br /><a href="https://github.com/dvlab-research/LargeScaleNeRFPytorch/issues?q=author%3AFanLu97" title="Bug reports">🐛</a></td>
      <td align="center"><a href="https://maybeshewill-cv.github.io"><img src="https://avatars.githubusercontent.com/u/15725187?v=4?s=100" width="100px;" alt=""/><br /><sub><b>MaybeShewill-CV</b></sub></a><br /><a href="https://github.com/dvlab-research/LargeScaleNeRFPytorch/issues?q=author%3AMaybeShewill-CV" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
