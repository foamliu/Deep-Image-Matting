import math
import os
import random
from random import shuffle

import cv2 as cv
import numpy as np
from keras.utils import Sequence

from config import batch_size
from config import fg_path, bg_path, a_path
from config import img_cols, img_rows
from config import unknown_code
from utils import safe_crop

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
with open('Combined_Dataset/Training_set/training_fg_names.txt') as f:
    fg_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_fg_names.txt') as f:
    fg_test_files = f.read().splitlines()
with open('Combined_Dataset/Training_set/training_bg_names.txt') as f:
    bg_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_bg_names.txt') as f:
    bg_test_files = f.read().splitlines()


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


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, a, fg, bg


def process(im_name, bg_name):
    im = cv.imread(fg_path + im_name)
    a = cv.imread(a_path + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_path + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    return composite4(im, bg, a, w, h)


def generate_trimap(alpha):
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    # fg = cv.erode(fg, kernel, iterations=np.random.randint(1, 3))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)


# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    y_indices, x_indices = np.where(trimap == unknown_code)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        filename = '{}_names.txt'.format(usage)
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()

        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 4), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols, 2), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i]
            fcount = int(name.split('.')[0].split('_')[0])
            bcount = int(name.split('.')[0].split('_')[1])
            im_name = fg_files[fcount]
            bg_name = bg_files[bcount]
            image, alpha, fg, bg = process(im_name, bg_name)

            # crop size 320:640:480 = 1:1:1
            different_sizes = [(320, 320), (480, 480), (640, 640)]
            crop_size = random.choice(different_sizes)

            trimap = generate_trimap(alpha)
            x, y = random_choice(trimap, crop_size)
            image = safe_crop(image, x, y, crop_size)
            alpha = safe_crop(alpha, x, y, crop_size)
            fg = safe_crop(fg, x, y, crop_size)
            bg = safe_crop(bg, x, y, crop_size)

            trimap = generate_trimap(alpha)

            # Flip array left to right randomly (prob=1:1)
            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                trimap = np.fliplr(trimap)
                alpha = np.fliplr(alpha)

            batch_x[i_batch, :, :, 0:3] = image / 255.
            batch_x[i_batch, :, :, 3] = trimap / 255.

            mask = np.equal(trimap, 128).astype(np.float32)
            batch_y[i_batch, :, :, 0] = alpha / 255.
            batch_y[i_batch, :, :, 1] = mask

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')


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

    from config import num_valid_samples
    valid_names = np.random.sample(names, num_valid_samples)
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
