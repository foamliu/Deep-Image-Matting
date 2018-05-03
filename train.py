import numpy as np
import os
import cv2 as cv
import keras
from keras.layers import Conv2D, UpSampling2D
import keras.backend as K
from vgg16 import vgg16_model

def load_data():
    # (num_samples, 224, 224, 3)
    x_train = np.empty((num_samples, 224, 224, 3), dtype=np.uint8)
    y_train = np.empty((num_samples, 224, 224), dtype=np.uint8)
    for i in range(num_samples):
        filename = os.path.join('data/test', '%05d.jpg' % (i + 1))
        bgr_img = cv.imread(filename)
        gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        x_train[i, :, :, :] = rgb_img
        y_train[i, :, :] = gray_img
    y_train = np.reshape(y_train, (num_samples, 224, 224, 1))
    return x_train, y_train


def matting_loss(y_true, y_pred):
    epsilon = 1e-6
    epsilon_sqr = epsilon**2
    return K.sum(K.sqrt(K.square(y_true - y_pred) + epsilon_sqr))


def encoder_decoder_model(img_rows, img_cols, channel=3):
    model = vgg16_model(img_rows, img_cols, channel, num_classes)

    # dense_1 = model.get_layer('dense_1')
    # flatten_1 = model.get_layer('flatten_1')

    model.layers.pop()  # dense_4
    model.layers.pop()  # dropout_2
    model.layers.pop()  # dense_2
    model.layers.pop()  # dropout_1
    model.layers.pop()  # dense_1
    model.layers.pop()  # flatten_1

    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv6'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5_1'))
    model.add(Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5_2'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4_1'))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4_2'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3_1'))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3_2'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2_1'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2_2'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1_1'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1_2'))
    model.add(Conv2D(1, (5, 5), activation='relu', padding='same', name='pred'))

    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    print(model.summary())

    model.compile(optimizer='adam', loss=matting_loss)

    return model


if __name__ == '__main__':
    img_rows, img_cols = 224, 224
    num_samples = 8041
    channel = 3
    num_classes = 10
    batch_size = 16
    epochs = 10
    train_data = 'data/test'

    x_train, y_train = load_data()

    # Load our model
    model = encoder_decoder_model(img_rows, img_cols, channel)

    # callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    callbacks = [tensor_board]

    model.fit(x_train,
              y_train,
              epochs=50,
              steps_per_epoch=num_samples / batch_size,
              callbacks=callbacks,
              verbose=1
    )
