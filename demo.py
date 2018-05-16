import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from data_generator import generate_trimap, get_top_left_corner, get_alpha_test
from model import create_model
from utils import get_final_output

if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 4

    model_weights_path = 'models/model.62-0.0524.hdf5'
    model = create_model()
    model.load_weights(model_weights_path)
    print(model.summary())

    out_test_path = 'merged_test/'
    test_images = [f for f in os.listdir(out_test_path) if
                   os.path.isfile(os.path.join(out_test_path, f)) and f.endswith('.png')]
    samples = random.sample(test_images, 10)

    for i in range(len(samples)):
        filename = samples[i]
        image_name = filename.split('.')[0]
        print('Start processing image: {}'.format(filename))
        x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
        bgr_img = cv.imread(os.path.join(out_test_path, filename))
        bg_h, bg_w = bgr_img.shape[:2]
        print(bg_h, bg_w)
        a = get_alpha_test(image_name)
        a_h, a_w = a.shape[:2]
        print(a_h, a_w)
        alpha = np.zeros((bg_h, bg_w), np.float32)
        alpha[0:a_h, 0:a_w] = a
        trimap = generate_trimap(alpha)
        x, y = get_top_left_corner(trimap)
        print(x, y)
        bgr_img = bgr_img[y:y + 320, x:x + 320]
        alpha = alpha[y:y + 320, x:x + 320]
        trimap = trimap[y:y + 320, x:x + 320]
        cv.imwrite('images/{}_image.png'.format(i), np.array(bgr_img).astype(np.uint8))
        cv.imwrite('images/{}_trimap.png'.format(i), np.array(trimap).astype(np.uint8))
        cv.imwrite('images/{}_alpha.png'.format(i), np.array(alpha).astype(np.uint8))

        x_test = np.empty((1, 320, 320, 4), dtype=np.float32)
        x_test[0, :, :, 0:3] = bgr_img / 255.
        x_test[0, :, :, 3] = trimap / 255.

        out = model.predict(x_test)
        out = np.reshape(out, (img_rows, img_cols))
        print(out.shape)
        out = out * 255.0
        out = get_final_output(out, trimap)
        out = out.astype(np.uint8)
        # cv.imshow('out', out)
        cv.imwrite('images/{}_out.png'.format(i), out)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    K.clear_session()
