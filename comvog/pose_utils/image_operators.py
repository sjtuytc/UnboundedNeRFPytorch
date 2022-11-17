import cv2
import pdb
import numpy as np
from scipy import stats


def get_bbox_from_img(image, color_thre=1e-2):
    # assuming the background is white, crop the center obj out.
    final_image = image.max() - image.max(-1)
    final_image[final_image < color_thre] = 0.0
    mask_image = final_image >= color_thre
    contours = cv2.findNonZero(final_image)
    contours = contours.squeeze()
    xmin, xmax = np.min(contours[:, 0]), np.max(contours[:, 0])
    ymin, ymax = np.min(contours[:, 1]), np.max(contours[:, 1])
    return mask_image, xmin, xmax, ymin, ymax
    

def change_background_from_black_to_white(image, color_thresh=1e-2):
    assert image.max() > 2, "the input image must be in (0-255) scale."
    image[image < color_thresh] = 255
    return image


def get_bbox_from_mask(label_img):
    contours = cv2.findNonZero(label_img)
    contours = contours.squeeze()
    xmin, xmax = np.min(contours[:, 0]), np.max(contours[:, 0])
    ymin, ymax = np.min(contours[:, 1]), np.max(contours[:, 1])
    return xmin, xmax, ymin, ymax
    

def apply_mask_on_img(one_img, label_img):
    assert one_img.max() > 2, "the input image must be in (0-255) scale."
    one_img[..., 0] = one_img[..., 0] * label_img + 255 * (1 - label_img)
    one_img[..., 1] = one_img[..., 1] * label_img + 255 * (1 - label_img)
    one_img[..., 2] = one_img[..., 2] * label_img + 255 * (1 - label_img)
    return one_img


def image_normalization_for_pose(image):
    assert image.max() > 2, "the input image must be in (0-255) scale."
    # image[:, : , 0] = (image[:, : , 0] - image[:, : , 0].mean()) / 255.0
    # image[:, : , 1] = (image[:, : , 1] - image[:, : , 1].mean()) / 255.0
    # image[:, : , 2] = (image[:, : , 2] - image[:, : , 2].mean()) / 255.0
    image = image / image.max()
    return image
