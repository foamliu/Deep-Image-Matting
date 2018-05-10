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


def load_data():
    pass


if __name__ == '__main__':
    alpha = cv.imread('mask/035A4301.jpg', 0)
    trimap = generate_trimap(alpha)
    cv.imshow('trimap', trimap)
    cv.waitKey(0)
