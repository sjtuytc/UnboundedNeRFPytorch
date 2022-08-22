import math
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import cycle
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union, Type
import pdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
import shutil
from mega_nerf.datasets.dataset_utils import get_rgb_index_mask
from mega_nerf.image_metadata import ImageMetadata
from mega_nerf.misc_utils import main_tqdm, main_print
from mega_nerf.ray_utils import get_ray_directions, get_rays, get_rays_batch

RAY_CHUNK_SIZE = 64 * 1024


class FilesystemDataset(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, chunk_paths: List[Path], num_chunks: int,
                 scale_factor: int, disk_flush_size: int):
        super(FilesystemDataset, self).__init__()
        self._device = device
        self._c2ws = torch.cat([x.c2w.unsqueeze(0) for x in metadata_items])
        self._near = near
        self._far = far
        self._ray_altitude_range = ray_altitude_range

        intrinsics = torch.cat(
            [torch.cat([torch.FloatTensor([x.W, x.H]), x.intrinsics]).unsqueeze(0) for x in metadata_items])
        if (intrinsics - intrinsics[0]).abs().max() == 0:
            main_print(
                'All intrinsics identical: W: {} H: {}, intrinsics: {}'.format(metadata_items[0].W, metadata_items[0].H,
                                                                               metadata_items[0].intrinsics))

            self._directions = get_ray_directions(metadata_items[0].W,
                                                  metadata_items[0].H,
                                                  metadata_items[0].intrinsics[0],
                                                  metadata_items[0].intrinsics[1],
                                                  metadata_items[0].intrinsics[2],
                                                  metadata_items[0].intrinsics[3],
                                                  center_pixels,
                                                  device).view(-1, 3)
        else:
            main_print('Differing intrinsics')
            self._directions = None
        parquet_paths = self._check_existing_paths(chunk_paths, center_pixels, scale_factor,
                                                   len(metadata_items))
        if parquet_paths is not None:
            main_print('Reusing {} chunks from previous run'.format(len(parquet_paths)))
            self._parquet_paths = parquet_paths
        else:
            self._parquet_paths = []
            self._write_chunks(metadata_items, center_pixels, device, chunk_paths, num_chunks, scale_factor,
                               disk_flush_size)

        self._parquet_paths.sort(key=lambda x: x.name)

        self._chunk_index = cycle(range(len(self._parquet_paths)))
        self._loaded_rgbs = None
        self._loaded_rays = None
        self._loaded_img_indices = None
        self._chunk_load_executor = ThreadPoolExecutor(max_workers=1)
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)
        self._chosen = None

    def load_chunk(self) -> None:
        chosen, self._loaded_rgbs, self._loaded_rays, self._loaded_img_indices = self._chunk_future.result()
        self._chosen = chosen
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)

    def get_state(self) -> str:
        return self._chosen

    def set_state(self, chosen: str) -> None:
        while self._chosen != chosen:
            self.load_chunk()

    def __len__(self) -> int:
        return self._loaded_rgbs.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'rgbs': self._loaded_rgbs[idx],
            'rays': self._loaded_rays[idx],
            'img_indices': self._loaded_img_indices[idx]
        }

    def _load_chunk_inner(self) -> Tuple[str, torch.FloatTensor, torch.FloatTensor, torch.ShortTensor]:
        if 'RANK' in os.environ:
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

        next_index = next(self._chunk_index)
        chosen = self._parquet_paths[next_index]
        loaded_chunk = pq.read_table(chosen)
        loaded_img_indices = torch.IntTensor(loaded_chunk['img_indices'].to_numpy().astype('int32'))

        if self._directions is not None:
            loaded_pixel_indices = torch.IntTensor(loaded_chunk['pixel_indices'].to_numpy())

            loaded_rays = []
            for i in range(0, loaded_pixel_indices.shape[0], RAY_CHUNK_SIZE):
                img_indices = loaded_img_indices[i:i + RAY_CHUNK_SIZE]
                unique_img_indices, inverse_img_indices = torch.unique(img_indices, return_inverse=True)
                c2ws = self._c2ws[unique_img_indices.long()].to(self._device)

                pixel_indices = loaded_pixel_indices[i:i + RAY_CHUNK_SIZE]
                unique_pixel_indices, inverse_pixel_indices = torch.unique(pixel_indices, return_inverse=True)

                # (#unique images, w*h, 8)
                image_rays = get_rays_batch(self._directions[unique_pixel_indices.long()],
                                            c2ws, self._near, self._far,
                                            self._ray_altitude_range).cpu()

                del c2ws

                loaded_rays.append(image_rays[inverse_img_indices, inverse_pixel_indices])

            loaded_rays = torch.cat(loaded_rays)
        else:
            loaded_rays = torch.FloatTensor(
                loaded_chunk.to_pandas()[['rays_{}'.format(i) for i in range(8)]].to_numpy())

        rgbs = torch.FloatTensor(loaded_chunk.to_pandas()[['rgbs_{}'.format(i) for i in range(3)]].to_numpy()) / 255.
        return str(chosen), rgbs, loaded_rays, loaded_img_indices

    def _write_chunks(self, metadata_items: List[ImageMetadata], center_pixels: bool, device: torch.device,
                      chunk_paths: List[Path], num_chunks: int, scale_factor: int, disk_flush_size: int) -> None:
        assert ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0

        path_frees = []
        total_free = 0

        for chunk_path in chunk_paths:
            chunk_path.mkdir(parents=True)

            _, _, free = shutil.disk_usage(chunk_path)
            total_free += free
            path_frees.append(free)

        parquet_writers = []

        index = 0

        max_index = max(metadata_items, key=lambda x: x.image_index).image_index
        if max_index <= np.iinfo(np.uint16).max:
            img_indices_dtype = np.uint16
        else:
            assert max_index <= np.iinfo(np.int32).max  # Can support int64 if need be
            img_indices_dtype = np.int32

        main_print('Max image index is {}: using dtype: {}'.format(max_index, img_indices_dtype))

        for chunk_path, path_free in zip(chunk_paths, path_frees):
            allocated = int(path_free / total_free * num_chunks)
            main_print('Allocating {} chunks to dataset path {}'.format(allocated, chunk_path))
            for j in range(allocated):
                parquet_path = chunk_path / '{0:06d}.parquet'.format(index)
                self._parquet_paths.append(parquet_path)

                dtypes = [('img_indices', pa.from_numpy_dtype(img_indices_dtype))]

                for i in range(3):
                    dtypes.append(('rgbs_{}'.format(i), pa.uint8()))

                if self._directions is not None:
                    dtypes.append(('pixel_indices', pa.int32()))
                else:
                    for i in range(8):
                        dtypes.append(('rays_{}'.format(i), pa.float32()))

                parquet_writers.append(pq.ParquetWriter(parquet_path, pa.schema(dtypes), compression='BROTLI'))

                index += 1

        main_print('{} chunks allocated'.format(index))

        write_futures = []
        rgbs = []
        rays = []
        indices = []
        in_memory_count = 0

        if self._directions is not None:
            all_pixel_indices = torch.arange(self._directions.shape[0], dtype=torch.int)

        with ThreadPoolExecutor(max_workers=len(parquet_writers)) as executor:
            for metadata_item in main_tqdm(metadata_items):
                image_data = get_rgb_index_mask(metadata_item)

                if image_data is None:
                    continue

                image_rgbs, img_indices, image_keep_mask = image_data
                rgbs.append(image_rgbs)
                indices.append(img_indices)
                in_memory_count += len(image_rgbs)

                if self._directions is not None:
                    image_pixel_indices = all_pixel_indices
                    if image_keep_mask is not None:
                        image_pixel_indices = image_pixel_indices[image_keep_mask == True]

                    rays.append(image_pixel_indices)
                else:
                    directions = get_ray_directions(metadata_item.W,
                                                    metadata_item.H,
                                                    metadata_item.intrinsics[0],
                                                    metadata_item.intrinsics[1],
                                                    metadata_item.intrinsics[2],
                                                    metadata_item.intrinsics[3],
                                                    center_pixels,
                                                    device)
                    image_rays = get_rays(directions, metadata_item.c2w.to(device), self._near, self._far,
                                          self._ray_altitude_range).view(-1, 8).cpu()

                    if image_keep_mask is not None:
                        image_rays = image_rays[image_keep_mask == True]

                    rays.append(image_rays)

                if in_memory_count >= disk_flush_size:
                    for write_future in write_futures:
                        write_future.result()

                    write_futures = self._write_to_disk(executor, torch.cat(rgbs), torch.cat(rays), torch.cat(indices),
                                                        parquet_writers, img_indices_dtype)

                    rgbs = []
                    rays = []
                    indices = []
                    in_memory_count = 0

            for write_future in write_futures:
                write_future.result()

            if in_memory_count > 0:
                write_futures = self._write_to_disk(executor, torch.cat(rgbs), torch.cat(rays), torch.cat(indices),
                                                    parquet_writers, img_indices_dtype)

                for write_future in write_futures:
                    write_future.result()
        for chunk_path in chunk_paths:
            chunk_metadata = {
                'images': len(metadata_items),
                'scale_factor': scale_factor
            }

            if self._directions is None:
                chunk_metadata['near'] = self._near
                chunk_metadata['far'] = self._far
                chunk_metadata['center_pixels'] = center_pixels
                chunk_metadata['ray_altitude_range'] = self._ray_altitude_range
            torch.save(chunk_metadata, chunk_path / 'metadata.pt')

        for parquet_writer in parquet_writers:
            parquet_writer.close()

        main_print('Finished writing chunks to dataset paths')

    def _check_existing_paths(self, chunk_paths: List[Path], center_pixels: bool, scale_factor: int, images: int) -> \
            Optional[List[Path]]:
        parquet_files = []

        num_exist = 0
        for chunk_path in chunk_paths:
            if chunk_path.exists():
                shutil.rmtree(chunk_path)  # clean the chunk path every exp time.
                # assert (chunk_path / 'metadata.pt').exists(), \
                #     "Could not find metadata file (did previous writing to this directory not complete successfully?)"
                # dataset_metadata = torch.load(chunk_path / 'metadata.pt', map_location='cpu')
                # assert dataset_metadata['images'] == images
                # assert dataset_metadata['scale_factor'] == scale_factor

                # if self._directions is None:
                #     assert dataset_metadata['near'] == self._near
                #     assert dataset_metadata['far'] == self._far
                #     assert dataset_metadata['center_pixels'] == center_pixels

                #     if self._ray_altitude_range is not None:
                #         assert (torch.allclose(torch.FloatTensor(dataset_metadata['ray_altitude_range']),
                #                                torch.FloatTensor(self._ray_altitude_range)))
                #     else:
                #         assert dataset_metadata['ray_altitude_range'] is None

                # for child in list(chunk_path.iterdir()):
                #     if child.name != 'metadata.pt':
                #         parquet_files.append(child)
                # num_exist += 1

        if num_exist > 0:
            assert num_exist == len(chunk_paths)
            return parquet_files
        else:
            return None

    def _write_to_disk(self, executor: ThreadPoolExecutor, rgbs: torch.Tensor, rays: torch.FloatTensor,
                       img_indices: torch.Tensor, parquet_writers: List[pq.ParquetWriter],
                       img_indices_dtype: Type[Union[np.ushort, np.int]]):
        indices = torch.randperm(rgbs.shape[0])
        shuffled_rgbs = rgbs[indices]
        shuffled_rays = rays[indices]
        shuffled_img_indices = img_indices[indices]

        num_chunks = len(parquet_writers)
        chunk_size = math.ceil(rgbs.shape[0] / num_chunks)

        futures = []

        def append(index: int) -> None:
            columns = {
                'img_indices': shuffled_img_indices[index * chunk_size:(index + 1) * chunk_size].numpy().astype(
                    img_indices_dtype)
            }

            for i in range(rgbs.shape[1]):
                columns['rgbs_{}'.format(i)] = shuffled_rgbs[index * chunk_size:(index + 1) * chunk_size, i].numpy()

            if self._directions is not None:
                columns['pixel_indices'] = shuffled_rays[index * chunk_size:(index + 1) * chunk_size].numpy()
            else:
                for i in range(rays.shape[1]):
                    columns['rays_{}'.format(i)] = shuffled_rays[index * chunk_size:(index + 1) * chunk_size, i].numpy()

            parquet_writers[index].write_table(pa.table(columns))

        for chunk_index in range(num_chunks):
            future = executor.submit(append, chunk_index)
            futures.append(future)

        return futures
