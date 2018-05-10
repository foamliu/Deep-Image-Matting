import keras.backend as K
import random
import numpy as np
import cv2 as cv
kernel = np.ones((5,5),np.uint8)

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
    a = np.sign(alpha) * 255
    dilation = cv.dilate(a, kernel, iterations=iter)
    trimap = np.sign(dilation - a) * 128 + a
    return trimap


def load_data():
    pass


if __name__ == '__main__':
    alpha = cv.imread('mask/1-1252426161dfXY.jpg', 0)
    trimap = generate_trimap(alpha)
    cv.imshow(trimap)
    cv.waitKey(0)

