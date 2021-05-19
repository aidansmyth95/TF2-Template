# Source: https://www.tensorflow.org/tensorboard/image_summaries
# TODO: test and debug
import io
import itertools
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf


class ConfusionMatrix():

    def __init__(self, model, file_writer_cm, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = list(map(str, range(0,10)))
        self.file_writer_cm = file_writer_cm
        self.confusion_matrix_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=self.log_confusion_matrix)


    def plot_to_image(self, figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    def log_confusion_matrix(self, epoch, logs):
        # Use the model to predict the values from the validation dataset.

        test_pred_raw = self.model.predict(self.X_val)
        test_pred     = np.argmax(test_pred_raw, axis=1)
        test_labels   = np.argmax(self.y_val, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
        # Log the confusion matrix as an image summary.
        figure = self.plot_confusion_matrix(cm, class_names=self.class_names)
        cm_image = self.plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)