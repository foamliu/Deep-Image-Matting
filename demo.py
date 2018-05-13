import os

import cv2 as cv
import keras.backend as K
import numpy as np

from data_generator import generate_trimap, get_crop_top_left, get_alpha
from new_start import autoencoder
from trimap_dict import trimap_init

if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 4
    trimap_init()

    model_weights_path = 'models/model.35-0.03.hdf5'
    model = autoencoder(img_rows, img_cols, channel)
    model.load_weights(model_weights_path)
    print(model.summary())

    for image_name in ['182_18293', '120_12081', '249_24980']:
        filename = '{}.png'.format(image_name)
        print('Start processing image: {}'.format(filename))
        x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
        bgr_img = cv.imread(os.path.join('merged', filename))
        bg_h, bg_w = bgr_img.shape[:2]
        print(bg_h, bg_w)
        a = get_alpha(image_name)
        a_h, a_w = a.shape[:2]
        print(a_h, a_w)
        alpha = np.zeros((bg_h, bg_w), np.float32)
        alpha[0:a_h, 0:a_w] = a
        trimap = generate_trimap(alpha)
        x, y = get_crop_top_left(trimap)
        print(x, y)
        bgr_img = bgr_img[y:y + 320, x:x + 320]
        alpha = alpha[y:y + 320, x:x + 320]
        trimap = trimap[y:y + 320, x:x + 320]
        cv.imwrite('images/{}_image.png'.format(image_name), np.array(bgr_img).astype(np.uint8))
        cv.imwrite('images/{}_trimap.png'.format(image_name), np.array(trimap).astype(np.uint8))
        cv.imwrite('images/{}_alpha.png'.format(image_name), np.array(alpha).astype(np.uint8))

        x_test = np.empty((1, 320, 320, 4), dtype=np.float32)
        x_test[0, :, :, 0:3] = bgr_img / 255.
        x_test[0, :, :, 3] = trimap / 255.

        out = model.predict(x_test)
        out = np.reshape(out, (img_rows, img_cols))
        print(out.shape)
        out = out * 255.0
        out = out.astype(np.uint8)
        # cv.imshow('out', out)
        cv.imwrite('images/{}_out.png'.format(image_name), out)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    K.clear_session()
