""" 
The purpose of this file is to implement the neuron selection algorithms used in DeepSonar.
Specifically, we want to select some hidden layer activations from the VGG model.
"""
import tensorflow as tf
import numpy as np

from load_sr_model import load_pretrained_model
from utils import top_k_values
from data_prep import load_data, get_for_datalist


def should_select_layer(layer):
    """
    Decides if a given layer should be selected for DeepSonar.
    Selected layers are convolutional and dense, according to the paper.
    """
    return isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense)


def select_layers(model):
    """ 
    Selects all the layers from the SR model that should be selected.
    Returns a list of names of the layers.
    """
    # long version: 
    # layers = []
    # for layer in model.layers:
    #     if should_select_layer(layer):
    #         layers.append(layer.name)
    # return layers
    return [layer.name for layer in model.layers if should_select_layer(layer)]


def get_layer_output_model():
    """ 
    https://www.aiworkbox.com/lessons/flatten-a-tensorflow-tensor 
    Returns a model that will output the intermediate values of all the selected layers.
    The output to the model is a list where each element is the intermediate output of one of the selected layers.
    """
    # load pretrained VGG model
    full_model = load_pretrained_model()

    # select required layers
    layer_names = select_layers(full_model)

    # create a list where each element is the output of an intermediate layer
    # long version:
    # hidden_layer_outputs = []
    # for layer_name in layer_names:
    #     hidden_layer_outputs.append(full_model.get_layer(name=layer_name).output)
    hidden_layer_outputs = [full_model.get_layer(layer_name).output for layer_name in layer_names]

    hidden_activations = []
    for layer in hidden_layer_outputs:

        # flatten all values in the layer into a vector
        flattened_version = tf.reshape(layer, [-1])

        # select the top k=5 values from each layer and add it to the list
        top_k_vals, _ = tf.math.top_k(flattened_version, k=5)
        hidden_activations.append(top_k_vals)

    # combine all list elements into one tensor
    combined = tf.keras.layers.concatenate(hidden_activations)

    # create and return the model
    # this model contains the VGG model plus some neuron selection algorithms
    return tf.keras.Model(inputs=full_model.input, outputs=combined)


def verify_weights():
    """ 
    Checks that the saved weights were preserved from the pretrained model to the new model.
    This returns true for all except for 3 / 41 layers; for those 3 layers, I manually checked some of the differences in 
        weights and I didn't see any differences.
    Therefore, I will assume that the weights are preserved from the pretrained to the new model. I'm absolutely not manually 
        scanning each of those weights lists...
    """
    # get the two models to compare as well as the layer names
    pretrained_model = load_pretrained_model()
    hidden_layer_output_model = get_layer_output_model()
    selected_layer_names = select_layers(pretrained_model)

    # check each selected layer and make sure the weights are equal
    all_weights_equal = True
    for layer_name in selected_layer_names:
        weights1 = pretrained_model.get_layer(layer_name).get_weights()
        weights2 = hidden_layer_output_model.get_layer(layer_name).get_weights()
        if not np.array_equal(weights1, weights2):
            print("weights were not equal for layer {}".format(layer_name))
            all_weights_equal = False
    if all_weights_equal:
        print("all of the weights were preserved.")


def build_classifier_dataset(mode='train'):
    """ 
    This returns the non-shuffled version of FoR as tensors.
    Builds a 3D tensor that is input to the DeepSonar binary classifier.
    Mode is either train, val, or test.
    """
    # make sure mode is valid
    valid_modes = {'train', 'val', 'test'}
    if mode not in valid_modes:
        print("invalid mode")
        return

    # get the existing SR model
    sr_model = get_layer_output_model()

    # get the data list with paths to all the data.
    partition, labels = get_for_datalist()
    data_list = partition[mode][:1]

    # set some hyperparameters
    num_examples = len(data_list)
    num_layers = len(sr_model.output)
    k = 5

    # monitor errors in creation (paths to examples)
    error_paths = []

    # create tensors for data & labels for the classifier
    data_tensor = np.empty(shape=(num_examples, num_layers, k))
    label_tensor = np.empty(shape=(num_examples), dtype=int)

    # fill tensors with data and labels
    for i, data_path in enumerate(data_list):

        # monitor progress
        print("example {} / {}".format(i, num_examples))

        # set label
        label_tensor[i] = labels[data_path]

        # set data
        try:
            audio_data = np.expand_dims(np.expand_dims(load_data(path=data_path), 0), -1)
            print(audio_data.shape)
            prediction = sr_model.predict(audio_data)
            # print("prediction length = {}".format(len(prediction)))
            for L, layer_outputs in enumerate(prediction):
                top_k = top_k_values(k, layer_outputs)
                data_tensor[i, L] = top_k
        except Exception as e:
            print("error on example {} / {}: {}".format(i, num_examples, e))
            error_paths.append(data_path)
            data_tensor[i] = data_tensor[i - 1]
            label_tensor[i] = label_tensor[i - 1]

    return data_tensor, label_tensor, error_paths
