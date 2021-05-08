""" A script to generate a Keras model to train on MNIST. This is a poor dummy model """
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Flatten, Reshape, Dense, Dropout
from tensorflow.keras.constraints import MinMaxNorm

def create_model(input_shape, num_classes):
    """ Create a model to train on MNIST data. """

    # ensure weights and bias values are clipped
    clip = MinMaxNorm(min_value=-0.5, max_value=0.5, axis=0)

    inp = Input(shape=input_shape)
    x = Reshape((28, 28, 1))(inp)
    x = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_constraint=clip, bias_constraint=clip)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_constraint=clip, bias_constraint=clip)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation='softmax', kernel_constraint=clip, bias_constraint=clip)(x)

    model = Model(inputs=inp, outputs=x)

    return model


def get_model_task(model):
    """ Get the task (classification or regression) from the model. Return True if classification. """
    found_act = None
    for layer in reversed(model.layers):
        if 'activation' in layer.get_config():
            found_act = layer.get_config()['activation']
            break
    if found_act is None:
        raise Exception('Uh oh, no activation found')
    
    # debatable whether we should be using relu, but oh well
    regression_act = ['linear', 'relu']

    # True if not regression act in last layer
    return not found_act in regression_act
