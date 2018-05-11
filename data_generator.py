import os
import random

import cv2 as cv
import numpy as np

from config import *
from trimap_dict import trimap_add

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
with open('Combined_Dataset/Training_set/training_fg_names.txt') as f:
    fg_files = f.read().splitlines()


def get_alpha(name):
    fg_i = int(name.split("_")[0])
    name = fg_files[fg_i]
    filename = os.path.join('mask', name)
    alpha = cv.imread(filename, 0)
    return alpha


def generate_trimap(alpha):
    iter = random.randint(1, 20)
    fg = alpha.copy()
    fg[alpha != 255] = 0
    unknown = alpha.copy()
    unknown[alpha != 0] = 255
    unknown = cv.dilate(unknown, kernel, iterations=iter)
    trimap = np.sign(unknown - fg) * 128 + fg
    return np.array(trimap).astype(np.uint8)


def get_crop_top_left(trimap):
    h, w = trimap.shape[:2]
    x, y = 0, 0
    for i in range(10):
        if w > img_cols:
            x = random.randint(0, w - img_cols)
        if h > img_rows:
            y = random.randint(0, h - img_rows)
        if trimap[y + img_rows / 2, x + img_cols / 2] == 128:
            break
    return x, y


def ensure_size(matrix, channel):
    h, w = matrix.shape[:2]
    if h >= img_rows and w >= img_cols:
        return matrix

    if channel > 1:
        ret = np.zeros((img_rows, img_cols, channel), dtype=np.float32)
        ret[0:h, 0:w, :] = matrix[:, :, :]
    else:
        ret = np.zeros((img_rows, img_cols), dtype=np.float32)
        ret[0:h, 0:w] = matrix[:, :]
    return ret


def data_gen(usage):
    filename = '{}_names.txt'.format(usage)
    with open(filename, 'r') as f:
        names = f.read().splitlines()
    i = 0
    while True:
        batch_x = np.empty((batch_size, img_rows, img_cols, 4), dtype=np.float32)
        batch_y = np.empty((batch_size, img_rows, img_cols, 1), dtype=np.float32)

        for i_batch in range(batch_size):
            name = names[i]
            filename = os.path.join('merged', name)
            bgr_img = cv.imread(filename)
            bg_h, bg_w = bgr_img.shape[:2]
            a = get_alpha(name)
            a_h, a_w = a.shape[:2]
            alpha = np.zeros((bg_h, bg_w), np.float32)
            alpha[0:a_h, 0:a_w] = a
            trimap = generate_trimap(alpha)
            x, y = get_crop_top_left(trimap)
            bgr_img = bgr_img[y:y + img_rows, x:x + img_cols]
            bgr_img = ensure_size(bgr_img, 3)
            trimap = trimap[y:y + img_rows, x:x + img_cols]
            trimap = ensure_size(trimap, 1)
            alpha = alpha[y:y + img_rows, x:x + img_cols]
            alpha = ensure_size(alpha, 1)
            batch_x[i_batch, :, :, 0:3] = bgr_img / 255.
            batch_x[i_batch, :, :, 3] = trimap / 255.
            batch_y[i_batch, :, :, 0] = alpha / 255.
            # store trimap
            trimap_add(alpha / 255., trimap)

            i += 1
            if i >= len(names):
                i = 0

        yield batch_x, batch_y


def train_gen():
    return data_gen('train')


def valid_gen():
    return data_gen('valid')


if __name__ == '__main__':
    filename = 'merged/357_35748.png'
    bgr_img = cv.imread(filename)
    bg_h, bg_w = bgr_img.shape[:2]
    print(bg_w, bg_h)
