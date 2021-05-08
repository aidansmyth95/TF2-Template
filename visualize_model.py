""" A script to visualize the model and what filters it has learned """
import os
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Model


def visualize_model(model, x_test, model_dir):
    """ Visualize the model and what filters it has learned """

    model_plot_dir = os.path.join(model_dir, 'model_plots')
    os.makedirs(model_plot_dir, exist_ok=True)

    input_data_sample = np.expand_dims(x_test[0], axis=0) # just one sample will do

    # create list of layers that we can use
    layer_list = []

    # summarize filter shapes
    for layer in model.layers:
        # check for convolutional layer
        if 'conv' in layer.name:
            layer_list.append(layer)
    del layer

    # for each applicable layer
    print('Visualizing the following layers:')
    for layer in layer_list:

        # get filter weights
        filters, biases = layer.get_weights()
        print('\t{} with shape {}'.format(layer.name, filters.shape))

        # normalize filter values to 0-1 so we can visualize them easier
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        # plot first n filters
        n_filters = 6
        ix = 1
        plt.clf()
        for i in range(n_filters):
            # get the filter
            f = filters[:, :, :, i]
            # plot each channel separately
            for j in range(f.shape[-1]):
                # specify subplot and turn of axis
                ax = plt.subplot(n_filters, f.shape[-1], ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(f[:, :, j], cmap='gray')
                ix += 1
        # show the figure
        plt.savefig(os.path.join(model_plot_dir, layer.name + '_' + str(n_filters) + '_filters.png'))

        # Activation maps, called feature maps, capture the result of applying the filters to input
        intermediate_model = Model(inputs=model.inputs, outputs=layer.output)
        # get feature map for first hidden layer
        feature_maps = intermediate_model.predict(input_data_sample)
        del intermediate_model
        # plot all as many feature maps in an NxN square
        square = int(np.sqrt(feature_maps.shape[-1]))
        ix = 1
        plt.clf()
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
                ix += 1
        # show the figure
        plt.savefig(os.path.join(model_plot_dir, layer.name + '_activation_feature_maps.png'))

    print('Model visualization complete.')