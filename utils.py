import multiprocessing

import keras.backend as K
import numpy as np
from tensorflow.python.client import device_lib


# simple alpha prediction loss
# def custom_loss(y_true, y_pred):
#     epsilon = 1e-6
#     epsilon_sqr = K.constant(epsilon ** 2)
#     return K.mean(K.sqrt(K.square(y_pred - y_true) + epsilon_sqr))


def custom_loss_wrapper(input_tensor):
    def custom_loss(y_true, y_pred):
        trimap = input_tensor[0, :, :, 3]
        mask = K.cast(K.equal(trimap, 128 / 255.), dtype='float32')
        diff = (y_pred - y_true) * mask
        num_pixels = K.sum(mask)
        epsilon = 1e-6
        epsilon_sqr = K.constant(epsilon ** 2)
        return K.sum(K.sqrt(K.square(diff) + epsilon_sqr)) / (num_pixels + epsilon)

    return custom_loss


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()
