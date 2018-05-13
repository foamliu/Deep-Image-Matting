import argparse

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.utils import multi_gpu_model

import migrate
from config import *
from data_generator import train_gen, valid_gen
from trimap_dict import trimap_init, trimap_clear
from utils import custom_loss, get_available_cpus, get_available_gpus

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--trained", help="path to save trained model files")
    args = vars(ap.parse_args())
    trained_path = args["trained"]
    if trained_path is None:
        trained_models_path = 'models/'
    else:
        # python train.py -t /mnt/Deep-Image-Matting/models/
        trained_models_path = '{}/'.format(trained_path)

    # Init trimap dict
    trimap_init()

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = trained_models_path + 'model.{epoch:02d}-{val_loss:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
    cleanup = LambdaCallback(on_epoch_begin=lambda epoch, logs: trimap_clear(epoch))


    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            self.model_to_save.save(model_names % (epoch, logs['val_loss']))


    # Load our model
    # Multi-GPUs
    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            print("Training with {} GPUs...".format(num_gpu))
            model = migrate.migrate_model(img_rows, img_cols, channel)
        new_model = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyCbk(model)
    else:
        new_model = migrate.migrate_model(img_rows, img_cols, channel)
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.99, nesterov=True)
    new_model.compile(optimizer='nadam', loss=custom_loss)

    print(new_model.summary())

    # Summarize then go!
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu/2))
    print('num_gpu={}\nnum_cpu={}\nworkers={}\ntrained_models_path={}.'.format(num_gpu, num_cpu, workers,
                                                                               trained_models_path))

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr, cleanup]

    # Start Fine-tuning
    new_model.fit_generator(train_gen(),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=valid_gen(),
                            validation_steps=num_valid_samples / batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            workers=workers
                            )
