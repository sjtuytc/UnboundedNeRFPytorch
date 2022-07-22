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

## Quick start on the mini dataset

## Data setup and preprocess
0. If you only want to run a demo and try out your scene reconstruction method, please download the mini version of Waymo Block-NeRF dataset provided by us. You do not need to run this procedure.
1. Download data from the official [Waymo](https://waymo.com/research/block-nerf/licensing/) website. 
2. Symbol link the downloaded dataset to the "data" folder. The Waymo official files (e.g., v1.0_waymo_block_nerf_mission_bay_train.tfrecord-00000-of-01063) would be put under the data folder.


## Installation
1. Create conda environment.
    ```bash
    conda create -n nerf-block python=3.7
    ```
2. Install tensorflow, our version: tensorflow with CUDA11.7.
    ```bash
    pip install tensorflow
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
