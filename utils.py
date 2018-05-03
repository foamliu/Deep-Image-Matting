import keras.backend as K


def matting_loss(y_true, y_pred):
    print('y_true.shape: ' + str(y_true.shape))
    print('y_pred.shape: ' + str(y_pred.shape))
    epsilon = 1e-6
    epsilon_sqr = K.constant(epsilon ** 2)
    return K.mean(K.sqrt(K.square(y_true - y_pred) + epsilon_sqr))

