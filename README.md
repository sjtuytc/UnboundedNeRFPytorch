# [CVPR22] Block-NeRF: Scalable Large Scene Neural View Synthesis

## Introduction

The [Block-NeRF project](https://waymo.com/intl/zh-cn/research/block-nerf/) builds the largest neural scene representation to date, capable of rendering an entire neighborhood of San Francisco. The abstract of their paper is as follows:

> We present Block-NeRF, a variant of Neural Radiance Fields that can represent large-scale environments. Specifically, we demonstrate that when scaling NeRF to render city-scale scenes spanning multiple blocks, it is vital to decompose the scene into individually trained NeRFs. This decomposition decouples rendering time from scene size, enables rendering to scale to arbitrarily large environments, and allows per-block updates of the environment. We adopt several architectural changes to make NeRF robust to data captured over months under different environmental conditions. We add appearance embeddings, learned pose refinement, and controllable exposure to each individual NeRF, and introduce a procedure for aligning appearance between adjacent NeRFs so that they can be seamlessly combined. We build a grid of Block-NeRFs from 2.8 million images to create the largest neural scene representation to date, capable of rendering an entire neighborhood of San Francisco.

The official results of Block-NeRF:

[![The demo of Block-NeRF](https://img.youtube.com/vi/6lGMCAzBzOQ/0.jpg)](https://www.youtube.com/watch?v=6lGMCAzBzOQ)

This project is the **non-official** implementation of Block-NeRF. Ideally, the features of this project would be:

- **PyTorch Implementation.** The official Block-NeRF paper uses tensorflow and requires TPUs. However, this implementation only needs PyTorch.
- **Better data preprocessing.** The original Block-NeRF paper requires downloading tons of data from Google Cloud Platform. This repo provide processed data and convenient scripts.
- **State-of-the-art performance.** This project produces state-of-the-art rendering quality with better efficiency.

- **Quick validation.** We provide quick validation tools to evaluate your ideas so that you don't need to train on the full Block-NeRF dataset.

- **Open research and better community.** Along with this project, we aim to developping a strong community working on this. We welcome you to joining us. The progress of this project would be updated at arxiv frequently.

Welcome to watch this project!

## News
- [2022.7.28] The data preprocess script is finished!
- [2022.7.20] This project started!

## Quick start on the Colab

This part is still under construction.

## Installation

1. Create conda environment.
   ```bash
   conda create -n nerf-block python=3.9
   ```
2. Install tensorflow and other libs. Our version: tensorflow with CUDA11.7.
   ```bash
   pip install tensorflow opencv-python matplotlib
   ```

## Data setup and preprocess

1. After signing the license on the [official waymo webiste](https://waymo.com/research/block-nerf/licensing/), download the Waymo Block dataset via the following command:

```bash
pip install gdown # download google drive download.
cd data
gdown --id 1iRqO4-GMqZAYFNvHLlBfjTcXY-l3qMN5 --no-cache
unzip v1.0.zip
cd ../
```

The Google cloud may [limit the download speed in this operation](https://stackoverflow.com/questions/16856102/google-drive-limit-number-of-download). Downloading in your browser can avoid this issue.
Alternatively, you can directly download from the official [Waymo](https://waymo.com/research/block-nerf/licensing/) website. However, this download may needs the sudo access to install the [gsutil tool](https://cloud.google.com/storage/docs/gsutil_install#deb) (if you don't have sudo access, you can download from your local laptop and then transport it to your server). The reference script is as follows:

```bash
# install gsutil tool
sudo apt-get install apt-transport-https ca-certificates gnupg # needs sudo access
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli # needs sudo access
gcloud init # login your google account then
cd data
gsutil -m cp -r \
  "gs://waymo-block-nerf/v1.0" \
  .
unzip v1.0.zip
cd ..
```

You may otherwise symbol link the downloaded dataset ("v1.0") under the "data" folder. The Waymo official files (e.g., v1.0/v1.0_waymo_block_nerf_mission_bay_train.tfrecord-00000-of-01063) would be put under the data folder.

2. Transfer the tensorflow version of data to the Pytorch version via the following command:

```bash
python data_preprocess/load_data.py
```

## Citations

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
