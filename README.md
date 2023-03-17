# Large-scale Neural Radiance Fields in Pytorch

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-6-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## 1. Introduction

**I was writing a paper about our progress now but feel free to use our code without citing us.**

This project aims for benchmarking several state-of-the-art large-scale radiance fields algorithms. We exchangely use terms "unbounded NeRF" and "large-scale NeRF" because we find the techniques behind them are closely related.

Instead of pursuing a big and complicated code system, we pursue a simple code repo with SOTA performance for unbounded NeRFs.

You are expected to get the following results in this repository:

| Benchmark                     | Methods      | PSNR      |
|-------------------------------|--------------|-----------|
| Unbounded Tanks & Temples     | NeRF++       | 20.49     |
| Unbounded Tanks & Temples     | Plenoxels    | 20.40     |
| Unbounded Tanks & Temples     | DVGO         | 20.10     |
| **Unbounded Tanks & Temples** | **Ours**     | **20.85** |
| Mip-NeRF-360 Benchmark          | NeRF         | 24.85     |
| Mip-NeRF-360 Benchmark          | NeRF++       | 26.21     |
| Mip-NeRF-360 Benchmark          | Mip-NeRF-360 | 28.94     |
| Mip-NeRF-360 Benchmark          | DVGO         | 25.42     |
| **Mip-NeRF-360 Benchmark**      | **Ours**     | **28.98** |

<details> 

<summary> Expand / collapse qualitative results. </summary>

## Tanks and Temples:

* Playground:

https://user-images.githubusercontent.com/31123348/220946729-d88db335-0618-4b75-9fc2-8de577e1ddb5.mp4

* Truck:

https://user-images.githubusercontent.com/31123348/220946857-0f4b7239-8be6-4fca-9bba-2f2425e857a5.mp4

* M60:

https://user-images.githubusercontent.com/31123348/220947063-068b94f6-3afb-421d-8746-43bcf9643a37.mp4

* Train:

https://user-images.githubusercontent.com/31123348/220947239-6528d542-b2b8-45e3-8e69-6e0eff869720.mp4

## Mip-NeRF-360 Benchmark:

* Bicycle:

https://user-images.githubusercontent.com/31123348/220947385-ab31c646-c671-4522-8e4f-a1982d98c753.mp4

* Stump:

https://user-images.githubusercontent.com/31123348/220947472-47dc4716-095b-45ec-890b-d6afd97de9e9.mp4

* Kitchen:

https://user-images.githubusercontent.com/31123348/220947597-68f7ec32-c761-4253-955a-a2acc6a2eb25.mp4

* Bonsai:

https://user-images.githubusercontent.com/31123348/220947686-d8957a2e-ef52-46cf-b437-28de91f55871.mp4

* Garden:

https://user-images.githubusercontent.com/31123348/220947771-bbd249c0-3d0b-4d25-9b79-d4de9af17c4a.mp4

* Counter:

https://user-images.githubusercontent.com/31123348/220947818-e5c6b07f-c930-48b2-8aa7-363182dea6be.mp4

* Room:

https://user-images.githubusercontent.com/31123348/220948025-25ce5cc1-3c9a-450c-920d-98a8f153a0fa.mp4

## San Francisco Mission Bay (dataset released by [Block-NeRF](https://waymo.com/research/block-nerf/)):
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

1. Clone this repository. Use depth == 1 to avoid download a large history.
   ```bash
   git clone --depth=1 git@github.com:sjtuytc/LargeScaleNeRFPytorch.git
   ```

2. Create conda environment.
   ```bash
   conda create -n large-scale-nerf python=3.9
   conda activate large-scale-nerf
   ```
3. Install pytorch, and other libs. Make sure your Pytorch version is compatible with your CUDA.
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

4. Install grid-based operators to avoid running them every time, cuda lib required. (Check via "nvcc -V" to ensure that you have a latest cuda.)
   ```bash
   apt-get install g++ build-essential  # ensure you have g++ and other build essentials, sudo access required.
   cd FourierGrid/cuda
   python setup.py install
   cd ../../
   ```
5. Install other libs used for reconstructing **custom** scenes. **This is only needed when you need to build your scenes.**
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

- **Disclaimer**: users are required to get permission from the original dataset provider. Any usage of the data must obey the license of the dataset owner.

(1) [Unbounded Tanks & Temples](https://www.tanksandtemples.org/). Download data from [here](https://drive.google.com/file/d/11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87/view). Then unzip the data.

```bash
cd data
gdown --id 11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87
unzip tanks_and_temples.zip
cd ../
```
	
(2) The [Mip-NeRF-360](https://jonbarron.info/mipnerf360/) dataset.

```bash
cd data
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
mkdir 360_v2
unzip 360_v2.zip -d 360_v2
cd ../
```

(3) [San Fran Cisco Mission Bay](https://waymo.com/research/block-nerf/).
What you should know before downloading the data:

- Our processed waymo data is significantly **smaller** than the original version (19.1GB vs. 191GB) because we store the camera poses instead of raw ray directions. Besides, our processed data is more friendly for Pytorch dataloaders. Download [the data](https://drive.google.com/drive/folders/1Lcc6MF35EnXGyUy0UZPkUx7SfeLsv8u9?usp=sharing) in the Google Drive. You may use [gdown](https://stackoverflow.com/questions/65001496/how-to-download-a-google-drive-folder-using-link-in-linux) to download the files via command lines. If you are interested in processing the raw waymo data on your own, please refer to [this doc](./docs/get_pytorch_waymo_dataset.md).

The downloaded data would look like this:

   ```
   data
      |
      |——————360_v2                                    // the root folder for the Mip-NeRF-360 benchmark
      |        └——————bicycle                          // one scene under the Mip-NeRF-360 benchmark
      |        |         └——————images                 // rgb images
      |        |         └——————images_2               // rgb images downscaled by 2
      |        |         └——————sparse                 // camera poses
      |        ...
      |——————tanks_and_temples                         // the root folder for Tanks&Temples
      |        └——————tat_intermediate_M60             // one scene under Tanks&Temples
      |        |         └——————camera_path            // render split camera poses, intrinsics and extrinsics
      |        |         └——————test                   // test split
      |        |         └——————train                  // train split
      |        |         └——————validation             // validation split
      |        ...
      |——————pytorch_waymo_dataset                     // the root folder for San Fran Cisco Mission Bay
      |        └——————cam_info.json                    // extracted cam2img information in dict.
      |        └——————coordinates.pt                   // global camera information used in Mega-NeRF, deprecated
      |        └——————train                            // train data
      |        |         └——————metadata               // meta data per image (camera information, etc)
      |        |         └——————rgbs                   // rgb images
      |        |         └——————split_block_train.json // split block informations
      |        |         └——————train_all_meta.json    // all meta informations in train folder
      |        └——————val                              // val data with the same structure as train
   ```
</details>

<details>
<summary> 4.2 Train models and see the results!</summary>

You only need to run "python run_FourierGrid.py" to finish the train-test-render cycle. Explanations of some arguments: 
```bash
--program: the program to run, normally --program train will be all you need.
--config: the config pointing to the scene file, e.g., --config FourierGrid/configs/tankstemple_unbounded/truck_single.py.
--num_per_block: number of blocks used in large-scale NeRFs, normally this is set to -1, unless specially needed.
--render_train: render the trained model on the train split.
--render_train: render the trained model on the test split.
--render_train: render the trained model on the render split.
--exp_id: add some experimental ids to identify different experiments. E.g., --exp_id 5.
--eval_ssim / eval_lpips_vgg: report SSIM / LPIPS(VGG) scores.
```

While we list major of the commands in scripts/train_FourierGrid.sh, we list some of commands below for better reproducibility.

```bash
# Unbounded tanks and temples
python run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/playground_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 57
python run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/train_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 12
python run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/truck_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 4
python run_FourierGrid.py --program train --config FourierGrid/configs/tankstemple_unbounded/m60_single.py --num_per_block -1 --render_train --render_test --render_video --exp_id 6

# 360 degree dataset
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/room_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 9
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/stump_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 10
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/bicycle_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 11
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/bonsai_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 3
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/garden_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/kitchen_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2
python run_FourierGrid.py --program train --config FourierGrid/configs/nerf_unbounded/counter_single.py --num_per_block -1 --eval_ssim --eval_lpips_vgg --render_train --render_test --render_video --exp_id 2

# San Francisco Mission Bay dataset
python run_FourierGrid.py --program train --config FourierGrid/configs/waymo/waymo_no_block.py --num_per_block 100 --render_video --exp_id 30
```

The old version of Block-NeRF is still provided to serve as a baseline, but it will be deprecated soon. We will mainly work on grid-based models later because they are simple and fast. Run the following command to reproduce the old Block-NeRF experiments:

```bash
bash scripts/block_nerf_train.sh
bash scripts/block_nerf_eval.sh
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
	python run_FourierGrid.py --config configs/custom/Madoka.py
	```
   You can replace configs/custom/Madoka.py by other configs.

4. Validating the training results to generate a fly-through video.

	```bash
	python run_FourierGrid.py --config configs/custom/Madoka.py --render_only --render_video --render_video_factor 8
	```
</details>


## 6. Citations & acknowledgements

Consider citing the following great works:

```
@inproceedings{dvgo,
  title={Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction},
  author={Sun, Cheng and Sun, Min and Chen, Hwann-Tzong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5459--5469},
  year={2022}
}

 @InProceedings{Tancik_2022_CVPR,
    author    = {Tancik, Matthew and Casser, Vincent and Yan, Xinchen and Pradhan, Sabeek and Mildenhall, Ben and Srinivasan, Pratul P. and Barron, Jonathan T. and Kretzschmar, Henrik},
    title     = {Block-NeRF: Scalable Large Scene Neural View Synthesis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {8248-8258}
}
```

We refer to the code and data from [DVGO](https://github.com/sunset1995/DirectVoxGO), [nerf-pl](https://github.com/kwea123/nerf_pl) and [SVOX2](https://github.com/sxyu/svox2), thanks for their great work!

## [Weekly classified NeRF](docs/weekly_nerf.md)
We track weekly NeRF papers and classify them. All previous published NeRF papers have been added to the list. We provide an [English version](docs/weekly_nerf.md) and a [Chinese version](docs/weekly_nerf_cn.md). We welcome [contributions and corrections](docs/contribute_weekly_nerf.md) via PR.

We also provide an [excel version](docs/weekly_nerf_meta_data.xlsx) (the meta data) of all NeRF papers, you can add your own comments or make your own paper analysis tools based on the structured meta data.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://sjtuytc.github.io/"><img src="https://avatars.githubusercontent.com/u/31123348?v=4?s=100" width="100px;" alt="Zelin Zhao"/><br /><sub><b>Zelin Zhao</b></sub></a><br /><a href="https://github.com/sjtuytc/LargeScaleNeRFPytorch/commits?author=sjtuytc" title="Code">💻</a> <a href="#maintenance-sjtuytc" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/SEUleaderYang"><img src="https://avatars.githubusercontent.com/u/55042050?v=4?s=100" width="100px;" alt="EZ-Yang"/><br /><sub><b>EZ-Yang</b></sub></a><br /><a href="https://github.com/sjtuytc/LargeScaleNeRFPytorch/commits?author=SEUleaderYang" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Alex-Alison-Zhang"><img src="https://avatars.githubusercontent.com/u/71915735?v=4?s=100" width="100px;" alt="Alex-Zhang"/><br /><sub><b>Alex-Zhang</b></sub></a><br /><a href="https://github.com/sjtuytc/LargeScaleNeRFPytorch/issues?q=author%3AAlex-Alison-Zhang" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://fanlu97.github.io/"><img src="https://avatars.githubusercontent.com/u/45007531?v=4?s=100" width="100px;" alt="Fan Lu"/><br /><sub><b>Fan Lu</b></sub></a><br /><a href="https://github.com/sjtuytc/LargeScaleNeRFPytorch/issues?q=author%3AFanLu97" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://maybeshewill-cv.github.io"><img src="https://avatars.githubusercontent.com/u/15725187?v=4?s=100" width="100px;" alt="MaybeShewill-CV"/><br /><sub><b>MaybeShewill-CV</b></sub></a><br /><a href="https://github.com/sjtuytc/LargeScaleNeRFPytorch/issues?q=author%3AMaybeShewill-CV" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/buer1121"><img src="https://avatars.githubusercontent.com/u/48516434?v=4?s=100" width="100px;" alt="buer1121"/><br /><sub><b>buer1121</b></sub></a><br /><a href="https://github.com/sjtuytc/LargeScaleNeRFPytorch/issues?q=author%3Abuer1121" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
