import random
import unittest

import cv2 as cv
import numpy as np
import os
from config import unknown_code
from data_generator import generate_trimap
from data_generator import get_alpha_test
from data_generator import random_choice
from utils import safe_crop


class TestStringMethods(unittest.TestCase):

    def test_generate_trimap(self):
        image = cv.imread('fg_test/cat-1288531_1920.png')
        alpha = cv.imread('mask_test/cat-1288531_1920.png', 0)
        trimap = generate_trimap(alpha)

        # ensure np.where works as expected.
        count = 0
        h, w = trimap.shape[:2]
        for i in range(h):
            for j in range(w):
                if trimap[i, j] == unknown_code:
                    count += 1
        x_indices, y_indices = np.where(trimap == unknown_code)
        num_unknowns = len(x_indices)
        self.assertEqual(count, num_unknowns)

        # ensure an unknown pixel is chosen
        ix = random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]

        self.assertEqual(trimap[center_x, center_y], unknown_code)

        x, y = random_choice(trimap)
        print(x, y)
        image = safe_crop(image, x, y)
        trimap = safe_crop(trimap, x, y)
        alpha = safe_crop(alpha, x, y)
        cv.imwrite('temp/test_generate_trimap_image.png', image)
        cv.imwrite('temp/test_generate_trimap_trimap.png', trimap)
        cv.imwrite('temp/test_generate_trimap_alpha.png', alpha)

    def test_flip(self):
        image = cv.imread('fg_test/cat-1288531_1920.png')
        alpha = cv.imread('mask_test/cat-1288531_1920.png', 0)
        trimap = generate_trimap(alpha)
        x, y = random_choice(trimap)
        image = safe_crop(image, x, y)
        trimap = safe_crop(trimap, x, y)
        alpha = safe_crop(alpha, x, y)
        image = np.fliplr(image)
        trimap = np.fliplr(trimap)
        alpha = np.fliplr(alpha)
        cv.imwrite('temp/test_flip_image.png', image)
        cv.imwrite('temp/test_flip_trimap.png', trimap)
        cv.imwrite('temp/test_flip_alpha.png', alpha)

    def test_different_sizes(self):
        different_sizes = [(320, 320), (320, 320), (320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)
        print('crop_size=' + str(crop_size))

    def test_resize(self):
        with open('Combined_Dataset/Test_set/test_bg_names.txt') as f:
            bg_test_files = f.read().splitlines()
        name = '35_716.png'
        filename = os.path.join('merged_test', name)
        image = cv.imread(filename)
        bg_h, bg_w = image.shape[:2]
        a = get_alpha_test(name)
        a_h, a_w = a.shape[:2]
        alpha = np.zeros((bg_h, bg_w), np.float32)
        alpha[0:a_h, 0:a_w] = a
        trimap = generate_trimap(alpha)
        # 剪切尺寸 320:640:480 = 3:1:1
        crop_size = (480, 480)
        x, y = random_choice(trimap, crop_size)
        image = safe_crop(image, x, y, crop_size)
        trimap = safe_crop(trimap, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)
        cv.imwrite('temp/test_resize_image.png', image)
        cv.imwrite('temp/test_resize_trimap.png', trimap)
        cv.imwrite('temp/test_resize_alpha.png', alpha)


if __name__ == '__main__':
    unittest.main()
