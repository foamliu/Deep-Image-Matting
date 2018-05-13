import argparse

import cv2 as cv
import numpy as np

from model import create_model
from trimap_dict import trimap_init

# python test.py -i "images/image.png" -t "images/trimap.png"
if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 4
    trimap_init()

    model_weights_path = 'models/model.35-0.03.hdf5'
    model = create_model(img_rows, img_cols, channel)
    model.load_weights(model_weights_path)
    print(model.summary())

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file")
    ap.add_argument("-t", "--trimap", help="path to the trimap file")
    args = vars(ap.parse_args())
    image_path = args["image"]
    trimap_path = args["trimap"]

    print('Start processing image: {}'.format(image_path))

    x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
    bgr_img = cv.imread(image_path)
    trimap = cv.imread(trimap_path, 0)

    x_test = np.empty((1, 320, 320, 4), dtype=np.float32)
    x_test[0, :, :, 0:3] = bgr_img / 255.
    x_test[0, :, :, 3] = trimap / 255.

    out = model.predict(x_test)
    out = np.reshape(out, (img_rows, img_cols))
    print(out.shape)
    out = out * 255.0
    out = out.astype(np.uint8)
    cv.imshow('out', out)
    cv.imwrite('images/out.png', out)
    cv.waitKey(0)
    cv.destroyAllWindows()
