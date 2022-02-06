import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

from data_prep import load_data
from neuron_selection import get_layer_output_model
from utils import top_k_values
from Labels import Labels

# goal of this file was to try to increase GPU util with tf.data
# ended up using the original DataGenerator instead
# this file is not currently used for anything
# followed part of tutorial: https://www.tensorflow.org/guide/data


def get_for_filenames(mode='training'):
    """
    Args: mode, either training, validation, or testing.
    Returns: a list of absolute file paths to data from specified partition as a tf.data Dataset.
    """
    glob_pattern = "/home/cameron/voice_data/fake-or-real/for-norm/{}/*/*".format(mode)
    return tf.data.Dataset.list_files(glob_pattern)


def process_audio_vgg(filename):
    """
    Returns: the processed audio and the label for each example, exactly how it was implemented by the VGG authors.
    """
    return np.expand_dims(np.expand_dims(load_data(filename.numpy()), 0), -1)


def process_audio_vgg_predict(filename):
    """
    Args:
        filename: absolute path to file
    Returns: the selected layer outputs from VGG.
    """
    # make prediction with intermediate layer output model (used in DeepSonar, not original VGG)
    vgg_model = get_layer_output_model()
    input_data_vgg = np.expand_dims(np.expand_dims(load_data(filename.numpy()), 0), -1)
    vgg_pred = vgg_model(input_data_vgg)

    # store the prediction in a tensor
    num_layers, k = 41, 5
    data = np.empty(shape=(num_layers, k))
    for L, layer_outputs in enumerate(vgg_pred):
        numeric_outputs = layer_outputs.numpy()
        top_k = top_k_values(k, numeric_outputs)
        data[L] = top_k

    # reshape the data to be fed into Dense layers
    dim0, dim1 = data.shape[0], data.shape[1]
    return np.reshape(data, newshape=(dim0 * dim1))


def tf_parse_audio(filename):
    # get data & label
    [data, ] = tf.py_function(process_audio_vgg, [filename], [tf.float32])
    label = tf.strings.split(filename, os.sep)[-2]

    # confirm data shape is correct (done in tutorials) & return
    required_shape = [1, 257, 250, 1]
    data = tf.ensure_shape(data, required_shape)
    return data, label


def tf_parse_audio_predict(filename):
    """
    Manually makes the prediction using VGG model.
    Does not combine VGG & binary classifier into a single model.
    Rather, predicts with an already trained VGG model.
    """
    # get model, data, & label
    [data, ] = tf.py_function(process_audio_vgg_predict, [filename], [tf.float32])
    label = tf.strings.split(filename, os.sep)[-2]

    # convert the label from string to integer
    if label == 'real':
        actual_label = Labels.REAL.value
    else:
        actual_label = Labels.FAKE.value

    # reshape the label so that the network trains
    # labels in DeepSonar dense network are of shape (None, 1)
    # this line of code has been a real pain in the ass.
    actual_label = tf.expand_dims(actual_label, -1)

    # confirm data shape is correct (done in tutorials) & return
    # data shape should now be the shape of the tensors being fed into DeepSonar, not VGG.
    # required_shape = [1, 257, 250, 1]
    # data = tf.ensure_shape(data, required_shape)

    # return data, label
    return data, actual_label
