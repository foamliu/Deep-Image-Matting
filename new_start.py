import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.utils.training_utils import multi_gpu_model

from encoder_decoder import build_encoder, build_decoder
from utils import do_compile
from utils import get_available_gpus


def do_autoencoder(img_rows, img_cols, channel=4):
    model = Sequential()
    # Encoder
    build_encoder(model, img_rows, img_cols, channel)
    # Decoder
    build_decoder(model)
    # Compile
    model = do_compile(model)
    return model


def autoencoder(img_rows, img_cols, channel=4):
    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            new_model = do_autoencoder(img_rows, img_cols, channel)
            new_model = multi_gpu_model(new_model, gpus=num_gpu)
    else:
        new_model = do_autoencoder(img_rows, img_cols, channel)

    new_model = do_compile(new_model)
    return new_model


if __name__ == '__main__':
    model = autoencoder(320, 320, 4)
    input_layer = model.get_layer('input')

    K.clear_session()
