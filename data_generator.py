import os
import random
from random import shuffle

import cv2 as cv
import numpy as np

from config import batch_size
from config import img_cols
from config import img_cols_half
from config import img_rows
from config import img_rows_half
from config import unknown
from utils import safe_crop

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
with open('Combined_Dataset/Training_set/training_fg_names.txt') as f:
    fg_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_fg_names.txt') as f:
    fg_test_files = f.read().splitlines()


def get_alpha(name):
    fg_i = int(name.split("_")[0])
    name = fg_files[fg_i]
    filename = os.path.join('mask', name)
    alpha = cv.imread(filename, 0)
    return alpha


def get_alpha_test(name):
    fg_i = int(name.split("_")[0])
    name = fg_test_files[fg_i]
    filename = os.path.join('mask_test', name)
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


# Randomly crop 320x320 (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap):
    y_indices, x_indices = np.where(trimap == unknown)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - img_cols_half)
        y = max(0, center_y - img_rows_half)
    return x, y


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
            image = cv.imread(filename)
            bg_h, bg_w = image.shape[:2]
            a = get_alpha(name)
            a_h, a_w = a.shape[:2]
            alpha = np.zeros((bg_h, bg_w), np.float32)
            alpha[0:a_h, 0:a_w] = a
            trimap = generate_trimap(alpha)
            x, y = random_choice(trimap)
            image = safe_crop(image, x, y)
            trimap = safe_crop(trimap, x, y)
            alpha = safe_crop(alpha, x, y)
            # 随机水平反转 (概率1:1)
            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                trimap = np.fliplr(trimap)
                alpha = np.fliplr(alpha)
            batch_x[i_batch, :, :, 0:3] = image / 255.
            batch_x[i_batch, :, :, 3] = trimap / 255.
            batch_y[i_batch, :, :, 0] = alpha / 255.

            i += 1
            if i >= len(names):
                i = 0

        yield batch_x, batch_y


def train_gen():
    return data_gen('train')


def valid_gen():
    return data_gen('valid')


def shuffle_data():
    num_fgs = 431
    num_bgs = 43100
    num_bgs_per_fg = 100
    num_valid_samples = 8620
    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    valid_names = random.sample(names, num_valid_samples)
    train_names = [n for n in names if n not in valid_names]
    shuffle(valid_names)
    shuffle(train_names)

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == '__main__':
    filename = 'merged/357_35748.png'
    bgr_img = cv.imread(filename)
    bg_h, bg_w = bgr_img.shape[:2]
    print(bg_w, bg_h)
