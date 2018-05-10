import keras.backend as K
import random
import numpy as np
import cv2 as cv

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))


# simple alpha prediction loss
def custom_loss(y_true, y_pred):
    epsilon = 1e-6
    epsilon_sqr = K.constant(epsilon ** 2)
    return K.mean(K.sqrt(K.square(y_pred - y_true) + epsilon_sqr))


def do_compile(model):
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.99, nesterov=True)
    model.compile(optimizer='nadam', loss=custom_loss)
    return model


def generate_trimap(alpha):
    iter = random.randint(1, 20)
    fg = alpha.copy()
    fg[alpha != 255] = 0
    unknown = alpha.copy()
    unknown[alpha != 0] = 255
    unknown = cv.dilate(unknown, kernel, iterations=iter)
    trimap = np.sign(unknown - fg) * 128 + fg
    return np.array(trimap).astype(np.uint8)


def data_gen(usage):
    pass


def train_gen():
    data_gen('train')


def valid_gen():
    data_gen('valid')


if __name__ == '__main__':
    pass
