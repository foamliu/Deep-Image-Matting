import argparse

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

import migrate
from config import patience, batch_size, epochs, num_train_samples, num_valid_samples
from data_generator import train_gen, valid_gen
from model import create_model
from utils import custom_loss_wrapper, get_available_cpus, get_available_gpus

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
        # model_checkpoint = MyCbk(model)
    else:
        if pretrained_path is not None:
            new_model = create_model()
            new_model.load_weights(pretrained_path)
        else:
            new_model = create_model()
            migrate.migrate_model(new_model)