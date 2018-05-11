import cv2 as cv
import numpy as np


def trimap_init():
    global trimap_dict
    trimap_dict = dict()
    global hit
    hit = 0
    global miss
    miss = 0


def trimap_add(alpha, trimap):
    global trimap_dict
    key = hash(str(alpha))
    trimap_dict[key] = trimap


def trimap_get(alpha):
    global hit
    global miss
    key = hash(str(alpha))
    if key in trimap_dict.keys():
        hit += 1
        return trimap_dict[key]
    else:
        miss += 1
        return None


def trimap_clear(epoch):
    global hit
    global miss
    size = len(trimap_dict)
    with open("training.txt", "a") as file:
        file.write("Epoch %d, cleaning %d trimaps, hit=%d, miss=%d." % (epoch, size, hit, miss))
    trimap_dict.clear()
    hit = 0
    miss = 0


if __name__ == '__main__':
    pass
