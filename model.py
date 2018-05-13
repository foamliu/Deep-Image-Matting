import keras.backend as K
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D
from keras.models import Sequential


def create_model(img_rows, img_cols, channel=4):
    model = Sequential()
    # Encoder
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel), name='input'))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Decoder
    # model.add(Conv2D(4096, (7, 7), activation='relu', padding='valid', name='conv6'))
    # model.add(BatchNormalization())
    # model.add(UpSampling2D(size=(7, 7)))

    model.add(
        Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv6', kernel_initializer='he_normal',
               bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    model.add(
        Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5', kernel_initializer='he_normal',
               bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    model.add(
        Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4', kernel_initializer='he_normal',
               bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    model.add(
        Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3', kernel_initializer='he_normal',
               bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    model.add(
        Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2', kernel_initializer='he_normal',
               bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    model.add(
        Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1', kernel_initializer='he_normal',
               bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(1, (5, 5), activation='sigmoid', padding='same', name='pred', kernel_initializer='he_normal',
                     bias_initializer='zeros'))
    return model


if __name__ == '__main__':
    model = create_model(320, 320, 4)
    input_layer = model.get_layer('input')

    K.clear_session()
