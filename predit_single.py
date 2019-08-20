# -*- coding: utf-8 -*-
import cv2 as cv
import keras.backend as K
import numpy as np

from model import build_encoder_decoder, build_refinement
from utils import get_final_output, create_patches, patch_dims, assemble_patches
import tensorflow as tf
import time

config = tf.ConfigProto(device_count = {"GPU": 1, "CPU": 1})
sess = tf.Session(config=config)
K.set_session(sess)

if __name__ == '__main__':
    # load network
    PATCH_SIZE = 320
    PRETRAINED_PATH = 'models/final.42-0.0398.hdf5'
    TRIMAP_PATH = "images/trimap2.png"
    IMG_PATH = "images/frame2.png"

    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(PRETRAINED_PATH)
    print(final.summary())

    # loading input files
    trimap = cv.imread(TRIMAP_PATH, cv.IMREAD_GRAYSCALE)
    img = cv.imread(IMG_PATH)
    result = np.zeros(trimap.shape, dtype=np.uint8)

    img_size = np.array(trimap.shape)

    # create patches
    x = np.dstack((img, np.expand_dims(trimap, axis=2))) / 255.
    patches = create_patches(x, PATCH_SIZE)

    # create mat for patches predictions
    patches_count = np.product(
        patch_dims(mat_size=trimap.shape, patch_size=PATCH_SIZE)
    )
    patches_predictions = np.zeros(shape=(patches_count, PATCH_SIZE, PATCH_SIZE))

    # predicting
    for i in range(patches.shape[0]):
        print("Predicting patches {}/{}".format(i + 1, patches_count))

        patch_prediction = final.predict(np.expand_dims(patches[i, :, :, :], axis=0))
        patches_predictions[i] = np.reshape(patch_prediction, (PATCH_SIZE, PATCH_SIZE)) * 255.

    # assemble
    result = assemble_patches(patches_predictions, trimap.shape, PATCH_SIZE)
    result = result[:img_size[0], :img_size[1]]

    prediction = get_final_output(result, trimap).astype(np.uint8)

    # save into files
    cv.imshow("result", prediction)
    cv.imshow("image", img)
    cv.waitKey(0)

    K.clear_session()

