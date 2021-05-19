""" A script to generate Keras callbacks """

import os
import time
import tensorflow as tf
from packaging import version
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from confusion_matrix_cbk import ConfusionMatrix
from misc import make_dir


def keras_callbacks(model_dir, model, X_val, y_val):
    """ A method to create lightweight Keras callbacks for training """

    # early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

    # Model Checkpoint for weights only. Note: TF2 uses accuracy instead of acc
    if version.parse(tf.__version__) < version.parse('2.0.0'):
        monitor = 'val_acc'
        model_filepath = os.path.join(model_dir,
                                    'epoch-{epoch:02d}_valacc-{val_acc:.6f}.hdf5')
    else:
        monitor = 'val_accuracy'
        model_filepath = os.path.join(model_dir,
                                    'epoch-{epoch:02d}_valacc-{val_accuracy:.6f}.hdf5') 
    checkpoint = ModelCheckpoint(model_filepath, monitor=monitor, verbose=1, save_best_only=True, mode='max')

    # TensorBoard logs
    log_dir = os.path.join(model_dir, 'logs')
    make_dir(log_dir)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # TensorBoard Confusion matrix callback
    file_writer_cm = tf.summary.create_file_writer(os.path.join(log_dir, 'confusion_matrix'))
    cm_callback_class = ConfusionMatrix(model, file_writer_cm, X_val, y_val)
    cm_callback = cm_callback_class.confusion_matrix_callback

    callbacks = [checkpoint, tensorboard_callback, cm_callback]

    return callbacks


