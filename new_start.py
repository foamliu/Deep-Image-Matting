import keras.backend as K
from keras.models import Sequential

from encoder_decoder import build_encoder, build_decoder
from utils import do_compile


def autoencoder(img_rows, img_cols, channel=4):
    model = Sequential()
    # Encoder
    build_encoder(model, img_rows, img_cols, channel)
    # Decoder
    build_decoder(model)
    # Compile
    do_compile(model)
    return model


if __name__ == '__main__':
    model = autoencoder(320, 320, 4)
    input_layer = model.get_layer('input')

    K.clear_session()
