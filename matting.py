from vgg16 import vgg16_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, \
    Reshape, Activation, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator


def matting_model(img_rows, img_cols, channel=3):
    model = vgg16_model(img_rows, img_cols, channel, num_classes)

    dense_1 = model.get_layer('dense_1')
    flatten_1 = model.get_layer('flatten_1')

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

    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model


if __name__ == '__main__':
    img_rows, img_cols = 224, 224
    num_train_samples = 1595
    channel = 3
    num_classes = 10
    batch_size = 16
    epochs = 10
    train_data = 'data/test'

    train_data_gen = ImageDataGenerator(rotation_range=20.,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
    train_generator = train_data_gen.flow_from_directory(train_data, (img_cols, img_rows), batch_size=batch_size,
                                                         class_mode=None)

    # Load our model
    model = matting_model(img_rows, img_cols, channel)

    model.fit_generator(train_generator,
                        steps_per_epoch=num_train_samples//batch_size,
                        epochs=50)
