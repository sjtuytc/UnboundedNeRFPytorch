import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import json
from kornia import create_meshgrid
from tqdm import tqdm
import pdb


def get_ray_directions(H, W, K):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = \
        torch.stack([(i - cx) / fx, -(j - cy) / fy, -
        torch.ones_like(i)], -1)  # (H, W, 3)
    # normalized camera space to pixel space
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions


def get_rays(directions, c2w):
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)

    # normalized before directions
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    return rays_o, rays_d


def find_idx_name(elements, img_name):
    #  image name to index
    for element in elements:
        if img_name in element:
            return element[1]
    return None

def find_nearest_idx(img_source,block_elements,train_meta):
    # fetch te closest image according to figure.
    cam_idx=img_source['cam_idx']
    distance = 1000
    img_idx=None
    img_nearest=None
    for element in block_elements:
        if train_meta[element[0]]['cam_idx']==cam_idx:  # if this is the same camera
            distance_temp = np.linalg.norm(np.array(img_source['origin_pos']) - np.array(train_meta[element[0]]['origin_pos']))
            if distance_temp < distance:
                distance = distance_temp
                img_idx=element[1]
                img_nearest=train_meta[element[0]]
    return img_idx


class WaymoDataset(Dataset):
    def __init__(self, root_dir, split='train', block='block_0',
                 img_downscale=4,
                 near=0.01, far=15,
                 test_img_name=None,
                 cam_begin=None,
                 cam_end=None):
        self.root_dir = root_dir
        self.split = split
        self.block = block
        self.img_downscale = img_downscale
        self.near = near
        self.far = far
        self.test_img_name = test_img_name
        
        # cam_begin is the start location of the compose phase
        self.cam_begin=cam_begin 
        self.cam_end=cam_end
        self.transform = transforms.ToTensor()
        self.read_json()

    def read_json(self):
        if self.split == "test" or self.split=="compose":  # test stage use the data in train.json
            # with open(os.path.join(self.root_dir, f'train/train_all_meta.json'), 'r') as fp:
            #     self.meta = json.load(fp)
            self.meta = torch.load(os.path.join(self.root_dir, f'train/train_all_meta.pt'))
            with open(os.path.join(self.root_dir, f'train/split_block_train.json'), 'r') as fp:
                self.block_split_info = json.load(fp)
        else:
            # with open(os.path.join(self.root_dir, f'{self.split}/{self.split}_all_meta.json'), 'r') as fp:
            #     self.meta = json.load(fp)
            self.meta = torch.load(os.path.join(self.root_dir, f'{self.split}/{self.split}_all_meta.pt'))
            with open(os.path.join(self.root_dir, f'{self.split}/split_block_{self.split}.json'), 'r') as fp:
                self.block_split_info = json.load(fp)

        if self.split == "train":
            self.image_path = []
            self.all_rays = []
            self.all_rgbs = []
            self.c2w = {}

            print("Loading the image...")

            for img_idx in tqdm(self.block_split_info[self.block]['elements']):
                '''
                img_idx:[image_name,index]
                '''
                img_info = self.meta[img_idx[0]]
                self.image_path.append(img_info['image_name'])
                exposure = torch.tensor(img_info['equivalent_exposure'])
                c2w = torch.FloatTensor(img_info['c2w'].float())
                self.c2w[img_idx[0]] = c2w

                width = img_info['W'] // self.img_downscale
                height = img_info['H'] // self.img_downscale

                # img = Image.open(os.path.join(
                #     self.root_dir, 'images', img_info['image_name'])).convert('RGB')
                img = Image.open(os.path.join(
                    self.root_dir, 'train', 'rgbs', img_info['image_name'] + ".png")).convert('RGB')
                if self.img_downscale != 1:
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                img = self.transform(img)  # (3,h,w)
                img = img.view(3, -1).permute(1, 0)
                self.all_rgbs.append(img)

                K = np.zeros((3, 3), dtype=np.float32)
                # fx=focal,fy=focal,cx=img_w/2,cy=img_h/2
                K[0, 0] = img_info['intrinsics'][0] // self.img_downscale
                K[1, 1] = img_info['intrinsics'][1] // self.img_downscale
                K[0, 2] = width * 0.5
                K[1, 2] = height * 0.5
                K[2, 2] = 1

                directions = get_ray_directions(height, width, K)
                rays_o, rays_d = get_rays(directions, c2w)

                # calculate the radius
                dx_1 = torch.sqrt(
                    torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :]) ** 2, -1))
                dx = torch.cat([dx_1, dx_1[-2:-1, :]], 0)
                radii = dx[..., None] * 2 / torch.sqrt(torch.tensor(12))
                '''
                rays_d = torch.tensor(
                    np.load(os.path.join(self.root_dir, "images", f"{img_idx[0]}_ray_dirs.npy"))
                )
                rays_o = torch.tensor(
                    np.load(os.path.join(self.root_dir, "images", f"{img_idx[0]}_ray_origins.npy"))
                )
                '''
                # rays_d has been normalized
                rays_d = rays_d.view(-1, 3)
                rays_o = rays_o.view(-1, 3)
                radii = radii.view(-1, 1)

                rays_t = img_idx[1] * torch.ones(len(rays_o), 1)

                self.all_rays.append(
                    torch.cat([
                        rays_o, rays_d,
                        radii,
                        exposure * torch.ones_like(rays_o[:, :1]),
                        self.near * torch.ones_like(rays_o[:, :1]),
                        self.far * torch.ones_like(rays_o[:, :1]),
                        rays_t], -1)
                )

            # ((N_images-1)*h*w, 8)
            self.all_rays = torch.cat(self.all_rays, 0)
            # ((N_images-1)*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)

            print("Has  totally loaded {0} images and {1} rays!".format(
                len(self.image_path), len(self.all_rays)))

        elif self.split == "test":
            self.N_frames = 10
            self.dy = np.linspace(0,0.2, self.N_frames)

        elif self.split == "compose": # input two views with distances
            print(f"Now is inferencing the images between {self.cam_begin} and {self.cam_end} ...")
            self.img_info = self.meta[self.cam_begin]
            self.img_info_end = self.meta[self.cam_end]

            origin_begin = self.img_info["origin_pos"]
            origin_end = self.img_info_end["origin_pos"]
            self.dx_dy_dz = np.array(origin_begin) - np.array(origin_end)
            print(f"The distance between {self.cam_begin} and {self.cam_end} is {self.dx_dy_dz}")

            if self.dx_dy_dz[1]<0.01:
                self.N_frames=1
            else:
                self.N_frames=self.dx_dy_dz[1]//0.01


    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            if len(self.block_split_info[self.block]) > 5:
                return 5
            else:
                return len(self.block_split_info[self.block])  # only validate 8 images

        return self.N_frames # test return the num of frames

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx, :10],
                      'rgbs': self.all_rgbs[idx],
                      'ts': self.all_rays[idx, 10].long()}
        elif self.split=="val":  # self.split == 'val':
            block_info = self.block_split_info[self.block]
            img_name, img_idx = block_info[idx]
            print("Basic image is {0}".format(img_name))
            img_info = self.meta[img_name]
            exposure = torch.tensor(img_info['equivalent_exposure'])

            c2w = torch.FloatTensor(img_info['c2w'].float())

            width = img_info['W'] // self.img_downscale
            height = img_info['H'] // self.img_downscale

            if self.split == 'val':
                img = Image.open(os.path.join(
                    self.root_dir, 'val', 'rgbs', img_info['image_name'] + ".png")).convert('RGB')
                if self.img_downscale != 1:
                    img = img.resize((width, height),
                                     Image.Resampling.LANCZOS)  # cv2.imshow("123.png",cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB)),cv2.waitKey()
                img = self.transform(img)  # (3,h,w)
                img = img.view(3, -1).permute(1, 0)

            self.K = {}
            K = np.zeros((3, 3), dtype=np.float32)
            # fx=focal,fy=focal,cx=img_w/2,cy=img_h/2
            K[0, 0] = img_info['intrinsics'][0] // self.img_downscale
            K[1, 1] = img_info['intrinsics'][1] // self.img_downscale
            K[0, 2] = width * 0.5
            K[1, 2] = height * 0.5
            K[2, 2] = 1

            directions = get_ray_directions(height, width, K)
            rays_o, rays_d = get_rays(directions, c2w)

            # calculate radius
            dx_1 = torch.sqrt(
                torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :]) ** 2, -1))
            dx = torch.cat([dx_1, dx_1[-2:-1, :]], 0)
            radii = dx[..., None] * 2 / torch.sqrt(torch.tensor(12))
            '''
            rays_d = torch.tensor(
                np.load(os.path.join(self.root_dir, "images", f"{img_idx[0]}_ray_dirs.npy"))
            )
            rays_o = torch.tensor(
                np.load(os.path.join(self.root_dir, "images", f"{img_idx[0]}_ray_origins.npy"))
            )
            '''
            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3)
            radii = radii.view(-1, 1)

            rays_t = img_idx * torch.ones(len(rays_o), 1)
            rays = torch.cat([
                rays_o, rays_d,
                radii,
                exposure * torch.ones_like(rays_o[:, :1]),
                self.near * torch.ones_like(rays_o[:, :1]),
                self.far * torch.ones_like(rays_o[:, :1]),
                rays_t], -1)
            sample = {"rays": rays[:, :10],
                      "ts": rays[:, 10].long(),
                      "w_h": [width, height]}
            sample["rgbs"]=img

        elif self.split=="test": #test
            img_info = self.meta[self.test_img_name]
            exposure = torch.tensor(img_info['equivalent_exposure'])
            c2w = torch.FloatTensor(img_info['c2w'])
            c2w[1, 3] += self.dy[idx]

            width = img_info['width'] // self.img_downscale
            height = img_info['height'] // self.img_downscale

            self.K = {}
            K = np.zeros((3, 3), dtype=np.float32)
            # fx=focal,fy=focal,cx=img_w/2,cy=img_h/2
            K[0, 0] = img_info['intrinsics'][0] // self.img_downscale
            K[1, 1] = img_info['intrinsics'][1] // self.img_downscale
            K[0, 2] = width * 0.5
            K[1, 2] = height * 0.5
            K[2, 2] = 1

            directions = get_ray_directions(height, width, K)
            rays_o, rays_d = get_rays(directions, c2w)

            # calculate radius
            dx_1 = torch.sqrt(
                torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :]) ** 2, -1))
            dx = torch.cat([dx_1, dx_1[-2:-1, :]], 0)
            radii = dx[..., None] * 2 / torch.sqrt(torch.tensor(12))

            # rays_d have been normalized
            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3)
            radii = radii.view(-1, 1)


            img_idx=find_idx_name(self.block_split_info[self.block]['elements'],self.test_img_name)

            if img_idx==None:
                print("It seems that the {0} doesn't belong to {1}".format(self.test_img_name,self.block))
                img_idx=find_nearest_idx(img_info,self.block_split_info[self.block]['elements'],self.meta)


            rays_t = img_idx * torch.ones(len(rays_o), 1)
            rays = torch.cat([
                rays_o, rays_d,
                radii,
                exposure * torch.ones_like(rays_o[:, :1]),
                self.near * torch.ones_like(rays_o[:, :1]),
                self.far * torch.ones_like(rays_o[:, :1]),
                rays_t], -1)
            sample = {"rays": rays[:, :10],
                      "ts": rays[:, 10].long(),
                      "w_h": [width, height]}


        else: #compose
            exposure = torch.tensor(self.img_info['equivalent_exposure'])
            c2w = self.img_info['c2w'].float()

            dx = np.linspace(0, self.dx_dy_dz[0], self.N_frames)
            dy = np.linspace(0, self.dx_dy_dz[1], self.N_frames)
            dz = np.linspace(0, self.dx_dy_dz[2], self.N_frames)

            c2w[0, 3] += dx[idx]
            c2w[1, 3] += dy[idx]
            c2w[2, 3] += dz[idx]

            width = self.img_info['W'] // self.img_downscale
            height = self.img_info['H'] // self.img_downscale

            self.K = {}
            K = np.zeros((3, 3), dtype=np.float32)
            # fx=focal,fy=focal,cx=img_w/2,cy=img_h/2
            K[0, 0] = self.img_info['intrinsics'][0] // self.img_downscale
            K[1, 1] = self.img_info['intrinsics'][1] // self.img_downscale
            K[0, 2] = width * 0.5
            K[1, 2] = height * 0.5
            K[2, 2] = 1

            directions = get_ray_directions(height, width, K)
            rays_o, rays_d = get_rays(directions, c2w)

            # calculare radius
            dx_1 = torch.sqrt(
                torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :]) ** 2, -1))
            dx = torch.cat([dx_1, dx_1[-2:-1, :]], 0)
            radii = dx[..., None] * 2 / torch.sqrt(torch.tensor(12))

            rays_d = rays_d.view(-1, 3)
            rays_o = rays_o.view(-1, 3)
            radii = radii.view(-1, 1)

            #   todo: change this because the default block is block 0
            img_idx = 0
            rays_t = img_idx * torch.ones(len(rays_o), 1)
            rays = torch.cat([
                rays_o, rays_d,
                radii,
                exposure * torch.ones_like(rays_o[:, :1]),
                self.near * torch.ones_like(rays_o[:, :1]),
                self.far * torch.ones_like(rays_o[:, :1]),
                rays_t], -1)
            sample = {"rays": rays[:, :10],
                      "ts": rays[:, 10].long(),
                      "w_h": [width, height]}


        return sample
