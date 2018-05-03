from vgg16 import vgg16_model
from keras.layers import Conv2D, UpSampling2D
from utils import matting_loss


def matting_model(img_rows, img_cols, channel=3):
    model = vgg16_model(img_rows, img_cols, channel)

    # dense_1 = model.get_layer('dense_1')
    # flatten_1 = model.get_layer('flatten_1')

    model.layers.pop()  # dense_4
    model.layers.pop()  # dropout_2
    model.layers.pop()  # dense_2
    model.layers.pop()  # dropout_1
    model.layers.pop()  # dense_1
    model.layers.pop()  # flatten_1

    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv6'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5_1'))
    model.add(Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5_2'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4_1'))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4_2'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3_1'))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3_2'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2_1'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2_2'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1_1'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1_2'))
    model.add(Conv2D(1, (5, 5), activation='relu', padding='same', name='pred'))

    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    print(model.summary())

    model.compile(optimizer='adam', loss=matting_loss)
    return model


if __name__ == '__main__':
    model = matting_model(224, 244, 3)
    model.save_weights('model_weights.h5')
