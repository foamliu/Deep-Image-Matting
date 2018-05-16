import random
import unittest

import cv2 as cv
import numpy as np

from config import unknown
from data_generator import generate_trimap
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
                if trimap[i, j] == unknown:
                    count += 1
        x_indices, y_indices = np.where(trimap == unknown)
        num_unknowns = len(x_indices)
        self.assertEqual(count, num_unknowns)

        # ensure an unknown pixel is chosen
        ix = random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]

        self.assertEqual(trimap[center_x, center_y], unknown)

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

    def test_split(self):
        pass


if __name__ == '__main__':
    unittest.main()
