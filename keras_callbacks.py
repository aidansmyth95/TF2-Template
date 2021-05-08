""" A script to generate Keras callbacks """

import os
import time
import tensorflow as tf
from packaging import version
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


def keras_callbacks():
    """ A method to create lightweight Keras callbacks for training """

    #TODO: add early stopping too

    # model dir to save model with timestamp
    time_str = time.strftime("model_%Y%m%d_%H%M%S")
    model_basedir = 'saved_model'
    model_dir = os.path.join(model_basedir, time_str)
    make_dir(model_basedir)
    make_dir(model_dir)
    # TF2 uses accuracy instead of acc
    if version.parse(tf.__version__) < version.parse('2.0.0'):
        monitor = 'val_acc'
        model_filepath = os.path.join(model_dir,
                                    'epoch-{epoch:02d}_valacc-{val_acc:.6f}.hdf5')
    else:
        monitor = 'val_accuracy'
        model_filepath = os.path.join(model_dir,
                                    'epoch-{epoch:02d}_valacc-{val_accuracy:.6f}.hdf5') 
    checkpoint = ModelCheckpoint(model_filepath, monitor=monitor, verbose=1, save_best_only=True, mode='max')

    # TensorBoard logs in same model_dir
    log_dir = os.path.join(model_dir, 'logs')
    make_dir(log_dir)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [checkpoint, tensorboard_callback]

    return model_dir, callbacks


def make_dir(dirname):
    """ A method to create a dir if it does not already exist """
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
