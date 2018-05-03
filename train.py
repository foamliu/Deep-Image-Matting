import numpy as np
import os
import cv2 as cv
import keras
from matting import matting_model


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
    model = matting_model(img_rows, img_cols, channel)
    model.load_weights('model_weights.h5')
    print(model.summary())

    # callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    callbacks = [tensor_board]

    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=50,
              callbacks=callbacks,
              verbose=1
              )
