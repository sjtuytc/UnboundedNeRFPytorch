import numpy as np
import torch
import torch.nn as nn
import imageio
import cv2
import pdb
import os

class ARF:
    def __init__(self, cfg, data_dict, device):
        super(ARF, self).__init__()
        assert 'arf' in cfg, "ARF should not be initialized according to cfg!"
        self.style_root = cfg.arf.style_root
        self.style_id = cfg.arf.style_id
        style_img_p = os.path.join(self.style_root, str(self.style_id) + ".jpg")
        self.device = device
        # assume the images are of the same height / width.
        self.w, self.h = data_dict['HW'][0][1], data_dict['HW'][0][0]

        # initialize style image.
        self.np_style_img = None
        self.style_img = self.load_style_img(style_img_p)
        
        
    def load_style_img(self, style_img_p):
        # resize style image such that its long side matches the long side of content images
        style_img = imageio.imread(style_img_p).astype(np.float32) / 255.0
        style_h, style_w = style_img.shape[:2]
        content_long_side = max([self.w, self.h])
        if style_h > style_w:
            style_img = cv2.resize(
                style_img,
                (int(content_long_side / style_h * style_w), content_long_side),
                interpolation=cv2.INTER_AREA,
            )
        else:
            style_img = cv2.resize(
                style_img,
                (content_long_side, int(content_long_side / style_w * style_h)),
                interpolation=cv2.INTER_AREA,
            )
        style_img = cv2.resize(
            style_img,
            (style_img.shape[1] // 2, style_img.shape[0] // 2),
            interpolation=cv2.INTER_AREA,
        )
        self.np_style_img = style_img
        style_img = torch.from_numpy(style_img).to(device=self.device)
        return style_img
        
    def match_colors_for_image_set(self, image_set, train_save_dir):  # code from ARF
        """
        image_set: [N, H, W, 3]
        style_img: [H, W, 3]
        """
        imageio.imwrite(
            os.path.join(train_save_dir, "style_image.png"),
            np.clip(self.np_style_img * 255.0, 0.0, 255.0).astype(np.uint8),
        )
        sh = image_set.shape
        image_set = image_set.reshape(-1, 3)
        image_set = torch.tensor(image_set).to(self.device)
        style_img = self.style_img.reshape(-1, 3).to(image_set.device)

        mu_c = image_set.mean(0, keepdim=True)
        mu_s = style_img.mean(0, keepdim=True)

        cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
        cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

        u_c, sig_c, _ = torch.svd(cov_c)
        u_s, sig_s, _ = torch.svd(cov_s)

        u_c_i = u_c.transpose(1, 0)
        u_s_i = u_s.transpose(1, 0)

        scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
        scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

        tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
        tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

        image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
        image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

        color_tf = torch.eye(4).float().to(tmp_mat.device)
        color_tf[:3, :3] = tmp_mat
        color_tf[:3, 3:4] = tmp_vec.T
        return image_set, color_tf

