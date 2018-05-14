import argparse

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

import migrate
from config import patience, batch_size, epochs, num_train_samples, num_valid_samples
from data_generator import train_gen, valid_gen
from model import create_model
from utils import custom_loss_wrapper, custom_loss, get_available_cpus, get_available_gpus

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", help="path to save checkpoint model files")
    ap.add_argument("-p", "--pretrained", help="path to save pretrained model files")
    args = vars(ap.parse_args())
    checkpoint_path = args["checkpoint"]
    pretrained_path = args["pretrained"]
    if checkpoint_path is None:
        checkpoint_models_path = 'models/'
    else:
        # python train.py -c /mnt/Deep-Image-Matting/models/
        checkpoint_models_path = '{}/'.format(checkpoint_path)

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

    class MyCbk(keras.callbacks.Callback):
        def __init__(self, model):
            keras.callbacks.Callback.__init__(self)
            self.model_to_save = model

        def on_epoch_end(self, epoch, logs=None):
            fmt = checkpoint_models_path + 'model.%02d-%.4f.hdf5'
            self.model_to_save.save(fmt % (epoch, logs['val_loss']))

    # Load our model, added support for Multi-GPUs
    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            if pretrained_path is not None:
                model = create_model()
                model.load_weights(pretrained_path)
            else:
                model = create_model()
                migrate.migrate_model(model)

        new_model = multi_gpu_model(model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = MyCbk(model)
    else:
        if pretrained_path is not None:
            new_model = create_model()
            new_model.load_weights(pretrained_path)
        else:
            new_model = create_model()
            migrate.migrate_model(new_model)

    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # new_model.compile(optimizer='nadam', loss=custom_loss_wrapper(new_model.get_input_at(0)))
    # new_model.compile(optimizer='nadam', loss=custom_loss)

    print(new_model.summary())

    # Summarize then go!
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 2))
    print('num_gpu={}\nnum_cpu={}\nworkers={}\ntrained_models_path={}.'.format(num_gpu, num_cpu, workers,
                                                                               checkpoint_models_path))

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    new_model.fit_generator(train_gen(),
                            steps_per_epoch=num_train_samples // batch_size,
                            validation_data=valid_gen(),
                            validation_steps=num_valid_samples // batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            use_multiprocessing=True,
                            workers=workers
                            )
