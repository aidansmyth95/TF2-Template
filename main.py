import os
import time
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from model import create_model, get_model_task
from keras_callbacks import keras_callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from packaging import version
import numpy as np
from visualize_results import visualize_results
from visualize_model import visualize_model
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from misc import make_dir


def main(args):

    # clear any existing TensorFlow Keras models
    K.clear_session()

    # print TF version
    print('\nTF version is {}'.format(tf.__version__))

    # print GPU info for TF2 only
    if version.parse(tf.__version__) >= version.parse('2.0.0'):
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) == 0:
            print('Warning: No GPUs detected')
        for gpu in gpus:
            print('Name {}\tType {}'.format(gpu.name, gpu.type))

    # parse args
    lr = args.lr
    train_fraction = args.train_fraction
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    loss = args.loss
    normalize_features = args.normalize_features

    # model dir to save model with timestamp
    time_str = time.strftime("model_%Y%m%d_%H%M%S")
    model_basedir = 'saved_model'
    model_dir = os.path.join(model_basedir, time_str)
    make_dir(model_basedir)
    make_dir(model_dir)

    # load data (MNIST) - seems to be (n_samples, 28, 28) this time
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='./mnist.npz')
    _, a, b = x_train.shape
    x_train = np.reshape(x_train, (x_train.shape[0], a * b))
    x_test = np.reshape(x_test, (x_test.shape[0], a * b))
    input_shape = list(x_train.shape)[1:] # exclude batch dim
    # make y data categorical
    # convert class vectors to binary class matrices
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # normalize the feature data
    if normalize_features:
        assert len(x_train.shape) <= 2, "Dim too large"
        print('Normalize training data...')
        min_max_scaler = MinMaxScaler()
        x_train = min_max_scaler.fit_transform(x_train)
        x_test = min_max_scaler.transform(x_test)
        joblib.dump(min_max_scaler, os.path.join(model_dir, 'scaler.save'))

    # Split the data
    print('Splitting training from validation data...')
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1-train_fraction, shuffle= True)

    # buid model
    model = create_model(input_shape=input_shape, num_classes=num_classes)
    print(model.summary())
    classification_bool = get_model_task(model) # True for classification, False for regression
    task = 'classification' if classification_bool else 'regression'
    print('Model task is {}.'.format(task))

    # compile model
    metrics = ['accuracy'] if classification_bool else ['mse']
    optimizer = Adam(lr=lr)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # generate Keras callbacks and timestamped model_dir
    callbacks = keras_callbacks(model_dir, model, x_val, y_val)

    # train model
    print('Training model...')
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1,
            shuffle=True, validation_data=(x_val, y_val), callbacks=callbacks)
    print('Model training complete.')

    # evaluate model
    eval_results = model.evaluate(x_test, y_test)
    print('Evaluation of test set results: {}'.format(eval_results))

    # test model and generate predictions
    print('Predicting results on test input data...')
    Y_pred = model.predict(x_test)
    print('Predictions generated.')

    # undo categorical encoding - makes it easier to plot
    Y_pred = np.argmax(Y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)    

    # visualize model performance
    visualize_results(y_test, Y_pred, classification_bool, model_dir)

    # visualize model, filters, etc
    visualize_model(model, x_test, model_dir)


if __name__ == '__main__':
    import argparse
    cmd_parser = argparse.ArgumentParser(description='Command line argument parser')
    cmd_parser.add_argument('--lr', type=float, default=0.0001)
    cmd_parser.add_argument('--train_fraction', type=float, default=0.8)
    cmd_parser.add_argument('--n_epochs', type=int, default=100)
    cmd_parser.add_argument('--batch_size', type=int, default=128)
    cmd_parser.add_argument('--loss', type=str, default='categorical_crossentropy')
    cmd_parser.add_argument('--normalize_features', type=bool, default=True)
    args = cmd_parser.parse_args()
    del cmd_parser

    main(args)
    print('Complete')