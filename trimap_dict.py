import cv2 as cv
import numpy as np


def trimap_init():
    global trimap_dict
    trimap_dict = dict()


def trimap_add(alpha, trimap):
    key = hash(str(alpha))
    trimap_dict[key] = trimap


def trimap_get(alpha):
    key = hash(str(alpha))
    return trimap_dict[key]


def trimap_clear():
    pass


if __name__ == '__main__':
    trimap_init()

    alpha = cv.imread('images/0_0_alpha.png')
    trimap = cv.imread('images/0_0_trimap.png')

    trimap_add(alpha, trimap)
    new_trimap = trimap_get(alpha)

    print(np.array_equal(trimap, new_trimap))
