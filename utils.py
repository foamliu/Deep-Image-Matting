import multiprocessing

import cv2 as cv
import keras.backend as K
import numpy as np
from tensorflow.python.client import device_lib

from config import epsilon, epsilon_sqr
from config import img_cols
from config import img_rows
from config import unknown_code


# overall loss: weighted summation of the two individual losses.
#
def overall_loss(y_true, y_pred):
    w_l = 0.5
    return w_l * alpha_prediction_loss(y_true, y_pred) + (1 - w_l) * compositional_loss(y_true, y_pred)


# alpha prediction loss: the abosolute difference between the ground truth alpha values and the
# predicted alpha values at each pixel. However, due to the non-differentiable property of
# absolute values, we use the following loss function to approximate it.
def alpha_prediction_loss(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    diff = y_pred[:, :, :, 0] - y_true[:, :, :, 0]
    diff = diff * mask
    num_pixels = K.sum(mask)
    return K.sum(K.sqrt(K.square(diff) + epsilon_sqr)) / (num_pixels + epsilon)


# compositional loss: the aboslute difference between the ground truth RGB colors and the predicted
# RGB colors composited by the ground truth foreground, the ground truth background and the predicted
# alpha mattes.
def compositional_loss(y_true, y_pred):
    mask = y_true[:, :, :, 1]
    mask = K.reshape(mask, (-1, img_rows, img_cols, 1))
    image = y_true[:, :, :, 2:5]
    fg = y_true[:, :, :, 5:8]
    bg = y_true[:, :, :, 8:11]
    c_g = image
    c_p = y_pred * fg + (1.0 - y_pred) * bg
    diff = c_p - c_g
    diff = diff * mask
    num_pixels = K.sum(mask)
    return K.sum(K.sqrt(K.square(diff) + epsilon_sqr)) / (num_pixels + epsilon)


# compute the MSE error given a prediction, a ground truth and a trimap.
# pred: the predicted alpha matte
# target: the ground truth alpha matte
# trimap: the given trimap
#
def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    # print('unknown: ' + str(unknown))
    loss = np.sum(np.square(error_map) * mask) / np.sum(mask)
    # print('mse_loss: ' + str(loss))
    return loss


# compute the SAD error given a prediction, a ground truth and a trimap.
#
def compute_sad_loss(pred, target, trimap):
    error_map = np.abs(pred - target) / 255.
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    loss = np.sum(error_map * mask)

    # the loss is scaled by 1000 due to the large images used in our experiment.
    loss = loss / 1000
    # print('sad_loss: ' + str(loss))
    return loss


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


def get_final_output(out, trimap):
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    return (1 - mask) * trimap + mask * out


def safe_crop(mat, x, y, crop_size=(img_rows, img_cols)):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (img_rows, img_cols):
        ret = cv.resize(ret, dsize=(img_rows, img_cols), interpolation=cv.INTER_NEAREST)
    return ret


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
