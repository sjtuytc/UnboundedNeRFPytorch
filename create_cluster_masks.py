import datetime
import os
import traceback
import zipfile
from argparse import Namespace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from mega_nerf.misc_utils import main_tqdm, main_print
from mega_nerf.opts import get_opts_base
from mega_nerf.ray_utils import get_ray_directions, get_rays
import pdb


def _get_mask_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--segmentation_path', type=str, default=None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--grid_dim', nargs='+', type=int, required=True)
    parser.add_argument('--ray_samples', type=int, default=1000)
    parser.add_argument('--ray_chunk_size', type=int, default=48 * 1024)
    parser.add_argument('--dist_chunk_size', type=int, default=64 * 1024 * 1024)
    parser.add_argument('--resume', default=False, action='store_true')

    return parser.parse_known_args()[0]


@record
@torch.inference_mode()
def main(hparams: Namespace) -> None:
    assert hparams.ray_altitude_range is not None
    output_path = Path(hparams.output)

    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, hours=24))
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        rank = int(os.environ['RANK'])
        if rank == 0:
            # output_path.mkdir(parents=True, exist_ok=hparams.resume)
            output_path.mkdir(parents=True, exist_ok=True) # set to true for the ease of development
        dist.barrier()
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        # output_path.mkdir(parents=True, exist_ok=hparams.resume)
        output_path.mkdir(parents=True, exist_ok=True) # set to true for the ease of development
        rank = 0
        world_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = Path(hparams.dataset_path)
    coordinate_info = torch.load(dataset_path / 'coordinates.pt', map_location='cpu')
    origin_drb = coordinate_info['origin_drb']
    pose_scale_factor = coordinate_info['pose_scale_factor']

    ray_altitude_range = [(x - origin_drb[0]) / pose_scale_factor for x in hparams.ray_altitude_range]

    metadata_paths = list((dataset_path / 'train' / 'metadata').iterdir()) \
                     + list((dataset_path / 'val' / 'metadata').iterdir())

    camera_positions = torch.cat([torch.load(x, map_location='cpu')['c2w'][:3, 3].unsqueeze(0) for x in metadata_paths])
    main_print('Number of images in dir: {}'.format(camera_positions.shape))

    min_position = camera_positions.min(dim=0)[0]
    max_position = camera_positions.max(dim=0)[0]

    main_print('Coord range: {} {}'.format(min_position, max_position))

    ranges = max_position[1:] - min_position[1:]
    offsets = [torch.arange(s) * ranges[i] / s + ranges[i] / (s * 2) for i, s in enumerate(hparams.grid_dim)]
    centroids = torch.stack((torch.zeros(hparams.grid_dim[0], hparams.grid_dim[1]),  # Ignore altitude dimension
                             torch.ones(hparams.grid_dim[0], hparams.grid_dim[1]) * min_position[1],
                             torch.ones(hparams.grid_dim[0], hparams.grid_dim[1]) * min_position[2])).permute(1, 2, 0)
    centroids[:, :, 1] += offsets[0].unsqueeze(1)
    centroids[:, :, 2] += offsets[1]
    centroids = centroids.view(-1, 3)

    main_print('Centroids shape: {}'.format(centroids.shape))

    near = hparams.near / pose_scale_factor

    if hparams.far is not None:
        far = hparams.far / pose_scale_factor
    else:
        far = 2

    torch.save({
        'origin_drb': origin_drb,
        'pose_scale_factor': pose_scale_factor,
        'ray_altitude_range': ray_altitude_range,
        'near': near,
        'far': far,
        'centroids': centroids,
        'grid_dim': (hparams.grid_dim),
        'min_position': min_position,
        'max_position': max_position,
        'cluster_2d': hparams.cluster_2d
    }, output_path / 'params.pt')

    z_steps = torch.linspace(0, 1, hparams.ray_samples, device=device)  # (N_samples)
    centroids = centroids.to(device)

    if rank == 0 and not hparams.resume:
        for i in range(centroids.shape[0]):
            (output_path / str(i)).mkdir(parents=True, exist_ok=True) # set to true for the ease of development

    if 'RANK' in os.environ:
        dist.barrier()

    cluster_dim_start = 1 if hparams.cluster_2d else 0
    for subdir in ['train', 'val']:
        metadata_paths = list((dataset_path / subdir / 'metadata').iterdir())
        for i in main_tqdm(np.arange(rank, len(metadata_paths), world_size)):
            metadata_path = metadata_paths[i]

            if hparams.resume:
                # Check to see if mask has been generated already
                all_valid = True
                filename = metadata_path.stem + '.pt'
                for j in range(centroids.shape[0]):
                    mask_path = output_path / str(j) / filename
                    if not mask_path.exists():
                        all_valid = False
                        break
                    else:
                        try:
                            with ZipFile(mask_path) as zf:
                                with zf.open(filename) as f:
                                    torch.load(f, map_location='cpu')
                        except:
                            traceback.print_exc()
                            all_valid = False
                            break

                if all_valid:
                    continue

            metadata = torch.load(metadata_path, map_location='cpu')

            c2w = metadata['c2w'].to(device)
            intrinsics = metadata['intrinsics']
            directions = get_ray_directions(metadata['W'],
                                            metadata['H'],
                                            intrinsics[0],
                                            intrinsics[1],
                                            intrinsics[2],
                                            intrinsics[3],
                                            hparams.center_pixels,
                                            device)

            rays = get_rays(directions, c2w, near, far, ray_altitude_range).view(-1, 8)
            min_dist_ratios = []
            for j in main_tqdm(range(0, rays.shape[0], hparams.ray_chunk_size)):
                rays_o = rays[j:j + hparams.ray_chunk_size, :3]
                rays_d = rays[j:j + hparams.ray_chunk_size, 3:6]
                near_bounds, far_bounds = rays[j:j + hparams.ray_chunk_size, 6:7], \
                                          rays[j:j + hparams.ray_chunk_size, 7:8]  # both (N_rays, 1)
                z_vals = near_bounds * (1 - z_steps) + far_bounds * z_steps

                xyz = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
                del rays_d
                del z_vals
                xyz = xyz.view(-1, 3)

                min_distances = []
                cluster_distances = []
                for k in range(0, xyz.shape[0], hparams.dist_chunk_size):
                    distances = torch.cdist(xyz[k:k + hparams.dist_chunk_size, cluster_dim_start:],
                                            centroids[:, cluster_dim_start:])
                    cluster_distances.append(distances)
                    min_distances.append(distances.min(dim=1)[0])

                del xyz

                cluster_distances = torch.cat(cluster_distances).view(rays_o.shape[0], -1,
                                                                      centroids.shape[0])  # (rays, samples, clusters)
                min_distances = torch.cat(min_distances).view(rays_o.shape[0], -1)  # (rays, samples)
                min_dist_ratio = (cluster_distances / (min_distances.unsqueeze(-1) + 1e-8)).min(dim=1)[0]
                del min_distances
                del cluster_distances
                del rays_o
                min_dist_ratios.append(min_dist_ratio)  # (rays, clusters)

            min_dist_ratios = torch.cat(min_dist_ratios).view(metadata['H'], metadata['W'], centroids.shape[0])

            filename = (metadata_path.stem + '.pt')

            if hparams.segmentation_path is not None:
                with ZipFile(Path(hparams.segmentation_path) / filename) as zf:
                    with zf.open(filename) as zf2:
                        segmentation_mask = torch.load(zf2, map_location='cpu')

            for j in range(centroids.shape[0]):
                cluster_ratios = min_dist_ratios[:, :, j]
                ray_in_cluster = cluster_ratios <= hparams.boundary_margin

                with ZipFile(output_path / str(j) / filename, compression=zipfile.ZIP_DEFLATED, mode='w') as zf:
                    with zf.open(filename, 'w') as f:
                        cluster_mask = ray_in_cluster.cpu()

                        if hparams.segmentation_path is not None:
                            cluster_mask = torch.logical_and(cluster_mask, segmentation_mask)

                        torch.save(cluster_mask, f)

                del ray_in_cluster


if __name__ == '__main__':
    main(_get_mask_opts())
