import argparse
from argparse import Namespace
from pathlib import Path
import cv2
import numpy as np
from importlib.metadata import MetadataPathFinder
import torch
from tqdm import tqdm
import pdb


def _get_images_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default="data/mega/ArtsQuad/ArtsQuad_dataset/images")
    parser.add_argument('--dataset_path', type=str, default="data/mega/quad/quad-pixsfm")

    return parser.parse_args()


def main(hparams: Namespace) -> None:
    image_path = Path(hparams.image_path)
    dataset_path = Path(hparams.dataset_path)
    (dataset_path / 'train' / 'rgbs').mkdir(exist_ok=True)
    (dataset_path / 'val' / 'rgbs').mkdir(exist_ok=True)

    with (Path(hparams.dataset_path) / 'mappings.txt').open() as f:
        lines = f.readlines()
    
    for line in tqdm(lines):
        image_name, metadata_name = line.strip().split(',')
        metadata_path = dataset_path / 'train' / 'metadata' / metadata_name
        if not metadata_path.exists():
            metadata_path = dataset_path / 'val' / 'metadata' / metadata_name
            assert metadata_path.exists()
        distorted = cv2.imread(str(image_path / image_name))
        metadata = torch.load(metadata_path, map_location='cpu')
        intrinsics = metadata['intrinsics']
        camera_matrix = np.array([[intrinsics[0], 0, intrinsics[2]],
                                    [0, intrinsics[1], intrinsics[3]],
                                    [0, 0, 1]])
        undistorted = cv2.undistort(distorted, camera_matrix, metadata['distortion'].numpy())
        assert undistorted.shape[0] == metadata['H']
        assert undistorted.shape[1] == metadata['W']
        save_p = str(metadata_path).replace("metadata", "rgbs").replace("pt", "jpg")
        cv2.imwrite(save_p, undistorted)


if __name__ == '__main__':
    main(_get_images_opts())