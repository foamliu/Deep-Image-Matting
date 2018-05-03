import keras.backend as K


def matting_loss(y_true, y_pred):
    epsilon = 1e-6
    epsilon_sqr = K.constant(epsilon ** 2)
    fake = K.constant(1)
    #return K.mean(K.sqrt(K.square(y_pred - y_true) + epsilon_sqr))
    return fake

