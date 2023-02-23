# [Weekly classified NeRF](docs/weekly_nerf.md)
We track weekly NeRF papers and classify them. All previous published NeRF papers have been added to the list. We provide an [English version](docs/weekly_nerf.md) and a [Chinese version](docs/weekly_nerf_cn.md). We welcome [contributions and corrections](docs/contribute_weekly_nerf.md) via PR.

We also provide an [excel version](docs/weekly_nerf_meta_data.xlsx) (the meta data) of all NeRF papers, you can add your own comments or make your own paper analysis tools based on the structured meta data.

# Large-scale Neural Radiance Fields in Pytorch

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## 1. Introduction

**I was writing a paper about our progress now but feel free to use our code without citing us.**

This project aims for benchmarking several state-of-the-art large-scale radiance fields algorithms. We exchangely use terms "unbounded NeRF" and "large-scale NeRF" because we find the techniques behind them are closely related.

Instead of pursuing a big and complicated code system, we pursue a simple code repo with SOTA performance for unbounded NeRFs.

You are expected to get the following results in this repository:



<details> 

<summary> Expand / collapse qualitative results. </summary>

## San Francisco Mission Bay (provided by [Block-NeRF](https://waymo.com/research/block-nerf/)):
* Training splits:

  https://user-images.githubusercontent.com/31123348/200509378-4b9fe63f-4fa4-40b1-83a9-b8950d981a3b.mp4

* Rotation: 

  https://user-images.githubusercontent.com/31123348/200509910-a5d8f820-143a-4e03-8221-b04d0db2d050.mov

</details>

Hope our efforts could help your research or projects!

## 2. News
- [2023.2.27] **A major update of our repository with better performance and full code release**. 

<details>
<summary> Expand / collapse older news. </summary>
	
- [2022.12.23] Released several weeks' NeRF. Too many papers pop out these days so the update speed is slow.
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
</details>

## 3. Installation
<details>
<summary>Expand / collapse installation steps.</summary>

1. Create conda environment.
   ```bash
   conda create -n large-scale-nerf python=3.9
   conda activate large-scale-nerf
   ```
2. Install pytorch, and other libs. Make sure your Pytorch version is compatible with your CUDA.
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   <!-- pip install tensorflow
   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html -->
   conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

3. Install grid-based operators to avoid running them every time, cuda lib required. (Check via "nvcc -V" to ensure that you have a latest cuda.)
   ```bash
   apt-get install g++ build-essential  # ensure you have g++ and other build essentials, sudo access required.
   cd comvog/cuda
   python setup.py install
   cd ../../
   ```
4. Install other libs used for reconstructing **custom** scenes. This is only needed when you need to build your scenes.
   ```bash
   sudo apt-get install colmap
   sudo apt-get install imagemagick  # required sudo accesss
   conda install pytorch-scatter -c pyg  # or install via https://github.com/rusty1s/pytorch_scatter
   ```
   You can use laptop version of COLMAP as well if you do not have access to sudo access on your server. However, we found if you do not set up COLMAP parameters properly, you would not get the SOTA performance.
</details>

## 4. Unbounded NeRF on the public datasets

Click the following sub-section titles to expand / collapse steps.

<details>
<summary> 4.1 Download processed data.</summary>

(1) [Unbounded Tanks & Temples](https://www.tanksandtemples.org/). Download data from [here](https://drive.google.com/file/d/11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87/view). Then unzip the data.

```bash
gdown --id 11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87
# Then unzip the data.
```
	
(2) The [Mip-NeRF-360](https://jonbarron.info/mipnerf360/) dataset.

```bash
cd data
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360
```

(3) San Fran Cisco Mission Bay.
What you should know before downloading the data:

- **Disclaimer**: you should ensure that you get the permission for usage from the original data provider. One should first sign the license on the [official waymo webiste](https://waymo.com/research/block-nerf/licensing/) to get the permission of downloading the Waymo data. Other data should be downloaded and used without obeying the original licenses.

- Our processed waymo data is significantly **smaller** than the original version (19.1GB vs. 191GB) because we store the camera poses instead of raw ray directions. Besides, our processed data is more friendly for Pytorch dataloaders. Download [the data](https://drive.google.com/drive/folders/1Lcc6MF35EnXGyUy0UZPkUx7SfeLsv8u9?usp=sharing) in the Google Drive. You may use [gdown](https://stackoverflow.com/questions/65001496/how-to-download-a-google-drive-folder-using-link-in-linux) to download the files via command lines.

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
