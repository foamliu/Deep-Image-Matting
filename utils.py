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
    alpha[alpha != 255] = 0
    dilation = cv.dilate(alpha, kernel, iterations=iter)
    trimap = (dilation - alpha) * 0.5 + alpha
    return np.array(trimap).astype(np.uint8)


def load_data():
    pass


if __name__ == '__main__':
    alpha = cv.imread('mask/1-1252426161dfXY.jpg', 0)
    trimap = generate_trimap(alpha)
    cv.imshow('trimap', trimap)
    cv.waitKey(0)
