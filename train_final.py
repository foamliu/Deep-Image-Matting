import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from config import patience, batch_size, epochs, num_train_samples, num_valid_samples
from data_generator import train_gen, valid_gen
from model import build_encoder_decoder, build_refinement
from utils import custom_loss, get_available_cpus, get_available_gpus

if __name__ == '__main__':
    checkpoint_models_path = 'models/'

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'final.{epoch:02d}-{val_loss:.4f}.hdf5'
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


    pretrained_path = 'models/final.61-0.0459.hdf5'
    # num_gpu = len(get_available_gpus())
    # if num_gpu >= 2:
    #     with tf.device("/cpu:0"):
    #         # Load our model, added support for Multi-GPUs
    #         encoder_decoder = build_encoder_decoder()
    #         final = build_refinement(encoder_decoder)
    #         final.load_weights(pretrained_path)
    #
    #     final = multi_gpu_model(final, gpus=num_gpu)
    #     # rewrite the callback: saving through the original model and not the multi-gpu model.
    #     model_checkpoint = MyCbk(final)
    # else:
    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(pretrained_path)

    # finetune the whole network together.
    for layer in final.layers:
        layer.trainable = True

    nadam = keras.optimizers.Nadam(lr=0.0002)
    # sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    decoder_target = tf.placeholder(dtype='float32', shape=(None, None, None, None))
    final.compile(optimizer=nadam, loss=custom_loss, target_tensors=[decoder_target])

    print(final.summary())

    # Summarize then go!
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 2))

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    final.fit_generator(train_gen(),
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_gen(),
                        validation_steps=num_valid_samples // batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=workers
                        )
