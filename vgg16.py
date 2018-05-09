# -*- coding: utf-8 -*-

from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

from utils import build_encoder


def vgg16_model(img_rows, img_cols, channel=3):
    model = Sequential()
    build_encoder(model, img_rows, img_cols, channel)

    # Add Fully Connected Layer
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='dense1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense2'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax', name='softmax'))

    # Loads ImageNet pre-trained data
    weights_path = 'models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path)

    return model
