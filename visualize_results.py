""" A script to visualize the results """
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import os

def visualize_results(Y_true, Y_pred, classification_task, model_dir):
    """ A method to visualize the results based on the task (classification or regression) """
    if classification_task:
        plot_confusion_matrix(Y_true, Y_pred, model_dir, 'confusion_matrix_plot.png')
    else:
        plot_cdf(Y_true, Y_pred, model_dir, 'cdf_plot.png')
        plot_scatterplot(Y_true, Y_pred, model_dir, 'scatter_plot.png')


#********************************************************
# Regression output visualizations
#********************************************************

def plot_cdf(Y_true, Y_pred, model_dir, filename, bins=1000):
    """ Plot Cumulative Distribution Frequency for a regression output """
    plt.clf()
    plt.figure(figsize=(16,10))
    
    # get histogram
    [aa, bb] = np.histogram(abs(Y_true - Y_pred), bins=bins)
    
    # semilogy plot
    plt.semilogy(0.5 * (bb[1:] + bb[:-1]),
                1 - np.cumsum(aa) / np.sum(aa),
                'g--',
                label='1st root')
    plt.grid(True)
    plt.title('CDF for error')
    plt.legend()
    plt.savefig(os.path.join(model_dir, filename))

#TODO: add true value line
def plot_scatterplot(Y_true, Y_pred, model_dir, filename, xmin=0, xmax=10, ymin=0, ymax=10):
    """ Plot scatter plot for predicted and true values """
    plt.clf()
    plt.figure(figsize=(10,10))

    plt.scatter(x=Y_true, y=Y_pred)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.grid(True)
    plt.title('Scatter plot')
    plt.savefig(os.path.join(model_dir, filename))


#********************************************************
# Classification output visualizations
#********************************************************

def plot_confusion_matrix(Y_true, Y_pred, model_dir, filename):
    """ Plot confusion matrix for two arrays of values """

    cm = confusion_matrix(Y_true, Y_pred)
    # assuming all classes are present in predictions and test data
    df_cm = pd.DataFrame(cm)
    
    plt.clf()
    plt.figure(figsize=(10,10))
    sns.heatmap(df_cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(model_dir, filename))
