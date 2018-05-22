import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from data_generator import generate_trimap, random_choice, get_alpha_test
from model import build_encoder_decoder, build_refinement
from utils import get_final_output, safe_crop

if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 4

    pretrained_path = 'models/final.25-0.0413.hdf5'
    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(pretrained_path)
    print(final.summary())

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
        different_sizes = [(320, 320), (320, 320), (320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)
        x, y = random_choice(trimap, crop_size)
        print(x, y)
        bgr_img = safe_crop(bgr_img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)
        trimap = safe_crop(trimap, x, y, crop_size)
        cv.imwrite('images/{}_image.png'.format(i), np.array(bgr_img).astype(np.uint8))
        cv.imwrite('images/{}_trimap.png'.format(i), np.array(trimap).astype(np.uint8))
        cv.imwrite('images/{}_alpha.png'.format(i), np.array(alpha).astype(np.uint8))

        x_test = np.empty((1, 320, 320, 4), dtype=np.float32)
        x_test[0, :, :, 0:3] = bgr_img / 255.
        x_test[0, :, :, 3] = trimap / 255.

        out = final.predict(x_test)
        out = np.reshape(out, (img_rows, img_cols))
        print(out.shape)
        out = out * 255.0
        # out = get_final_output(out, trimap)
        out = out.astype(np.uint8)
        # cv.imshow('out', out)
        cv.imwrite('images/{}_out.png'.format(i), out)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    K.clear_session()
