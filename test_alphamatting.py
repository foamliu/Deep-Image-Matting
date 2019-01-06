import os

import cv2 as cv
import numpy as np

from model import build_encoder_decoder, build_refinement

if __name__ == '__main__':
    pretrained_path = 'models/final.42-0.0398.hdf5'
    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(pretrained_path)
    print(final.summary())

    images = [f for f in os.listdir('alphamatting/input_lowres') if f.endswith('.png')]

    for image_name in images:
        filename = os.path.join('alphamatting/input_lowres', image_name)
        im = cv.imread(filename)
        im_h, im_w = im.shape[:2]

        for id in [1, 2, 3]:
            trimap_name = os.path.join('alphamatting/trimap_lowres/Trimap{}'.format(id), image_name)
            trimap = cv.imread(trimap_name, 0)

            for i in range(0, np.ceil(im_h / 320)):
                for j in range(0, np.ceil(im_w / 320)):
                    x = j * 320
                    y = i * 320
                    w = min(320, im_w - x)
                    h = min(320, im_h - y)
                    im_crop = im[y:y + h, x:x + w]
                    tri_crop = trimap[y:y + h, x:x + w]
