import argparse

import cv2 as cv
import numpy as np

from model import build_encoder_decoder, build_refinement
from utils import get_final_output

# python test.py -i "images/image.png" -t "images/trimap.png"
if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 4

    model_weights_path = 'models/final.42-0.0398.hdf5'
    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(model_weights_path)
    print(final.summary())

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file")
    ap.add_argument("-t", "--trimap", help="path to the trimap file")
    args = vars(ap.parse_args())
    image_path = args["image"]
    trimap_path = args["trimap"]

    if image_path is None:
        image_path = 'images/image.jpg'
    if trimap_path is None:
        trimap_path = 'images/trimap.jpg'

    print('Start processing image: {}'.format(image_path))

    x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
    bgr_img = cv.imread(image_path)
    trimap = cv.imread(trimap_path, 0)

    x_test[0, :, :, 0:3] = bgr_img / 255.
    x_test[0, :, :, 3] = trimap / 255.

    out = final.predict(x_test)
    out = np.reshape(out, (img_rows, img_cols))
    print(out.shape)
    out = out * 255.0
    out = get_final_output(out, trimap)
    out = out.astype(np.uint8)
    cv.imshow('out', out)
    cv.imwrite('images/out.png', out)
    cv.waitKey(0)
    cv.destroyAllWindows()
