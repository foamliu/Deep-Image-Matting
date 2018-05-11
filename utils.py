import keras.backend as K
from tensorflow.python.client import device_lib

from trimap_dict import trimap_get


# simple alpha prediction loss
def custom_loss(y_true, y_pred):
    trimap = trimap_get(y_true)
    trimap[trimap != 128] = 0
    trimap[trimap == 128] = 1
    diff = (y_pred - y_true) * trimap
    num_pixels = K.sum(trimap)
    epsilon = 1e-6
    epsilon_sqr = K.constant(epsilon ** 2)
    return K.sum(K.sqrt(K.square(diff) + epsilon_sqr)) / num_pixels


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def do_compile(model):
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.99, nesterov=True)
    model.compile(optimizer='nadam', loss=custom_loss)
    return model
