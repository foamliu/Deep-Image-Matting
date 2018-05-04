import numpy as np
import os
import cv2 as cv
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matting
import transfer
from console_progressbar import ProgressBar


def load_data():
    # (num_samples, 224, 224, 3)
    num_samples = 8146
    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    num_valid = num_samples - num_train
    pb = ProgressBar(total=100, prefix='Loading data', suffix='', decimals=3, length=50, fill='=')

    x_train = np.empty((num_train, 224, 224, 3), dtype=np.float32)
    y_train = np.empty((num_train, 224, 224, 1), dtype=np.float32)
    x_valid = np.empty((num_valid, 224, 224, 3), dtype=np.float32)
    y_valid = np.empty((num_valid, 224, 224, 1), dtype=np.float32)

    i_train = i_valid = 0
    for root, dirs, files in os.walk("data", topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            bgr_img = cv.imread(filename)
            gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            if (filename.startswith('data/train')):
                x_train[i_train, :, :, :] = rgb_img / 255.
                y_train[i_train, :, :, 0] = gray_img / 255.
                i_train += 1
            else:
                x_valid[i_valid, :, :, :] = rgb_img / 255.
                y_valid[i_valid, :, :, 0] = gray_img / 255.
                i_valid += 1

            i = i_train + i_valid
            if i % batch_size == 0:
                pb.print_progress_bar((i + 2) * 100 / num_samples)
    return x_train, y_train, x_valid, y_valid


if __name__ == '__main__':
    img_rows, img_cols = 224, 224
    num_samples = 8041
    channel = 3
    num_classes = 10
    batch_size = 16
    epochs = 1000
    train_data = 'data/test'
    patience = 50

    # Load our model
    model_path = 'model_weights.h5'
    if os.path.exists(model_path):
        model = matting.matting_model(img_rows, img_cols, channel)
        model.load_weights(model_path)
    else:
        model = transfer.matting_model(img_rows, img_cols, channel)

    print(model.summary())

    # Load our data
    x_train, y_train, x_valid, y_valid = load_data()

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    trained_models_path = 'models/model'
    model_names = trained_models_path + '.{epoch:02d}-{val_loss:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    model.fit(x_train,
              y_train,
              validation_data=(x_valid, y_valid),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              verbose=1
              )
