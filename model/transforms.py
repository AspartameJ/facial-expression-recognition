# The implementation is based on Facial-Expression-Recognition, available at
# https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
import numbers
import types

import numpy as np
import cv2


def to_ndarray(img):
    # put it from HWC to CHW format
    img = img.transpose(2, 0, 1)
    if isinstance(img, np.ndarray):
        return img.astype("float32")/255
    else:
        return img


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = img.shape[:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[i:i+th, j:j+tw]


def five_crop(img, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(
            size) == 2, 'Please provide only two dimensions (h, w) for size.'

    h, w = img.shape[:2]
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError(
            'Requested crop size {} is bigger than input size {}'.format(
                size, (h, w)))
    tl = img[0:crop_h, 0:crop_w]
    tr = img[0:crop_h, w-crop_w:w]
    bl = img[h-crop_h:h, 0:crop_w]
    br = img[h-crop_h:h, w-crop_w:w]
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)


class TenCrop(object):

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(
                size
            ) == 2, 'Please provide only two dimensions (h, w) for size.'
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        first_five = five_crop(img, self.size)

        if self.vertical_flip:
            img = cv2.flip(img, 0)  # flip vertically
        else:
            img = cv2.flip(img, 1)  # flip horizontally

        second_five = five_crop(img, self.size)

        return first_five + second_five


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToNdarray(object):

    def __call__(self, pic):
        return to_ndarray(pic)


class Lambda(object):

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

