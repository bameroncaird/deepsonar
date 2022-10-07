"""
In this file, we will take care of the full DeepSonar model architecture.
This is the model discussed in the DeepSonar paper. The implementation was not shared by the authors.
See README.md for more information about this model.
"""

import tensorflow as tf

# get access to the underlying SR system
from vgg_sr_model import load_pretrained_vgg_model


def should_select_layer(layer):
    """
    Decides if a given layer should be selected for DeepSonar.
    Selected layers are convolutional and dense (FC), according to the paper's implementation section.
    """
    return isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense)

def select_layers(model):
    """ 
    Selects all the layers from an input model that should be selected.
    Returns a list of names of the layers.
    """
    return [layer.name for layer in model.layers if should_select_layer(layer)]

def top_k_activated_neuron(layer_values):
    """
    Implements the top-k activated neuron (TKAN) algorithm discussed in the DeepSonar paper.
    This algorithm selects the hidden layer activations from the SR model for the classifier.
    See equation 3 in the paper for the details of TKAN.
    For this function, we will operate on a single layer.
    """
    # we don't care about the layer's shape anymore, we just want to select the top 5 values
    # even if the layer is the shape of some tensor (e.g. Conv2D), we will flatten it to compare the values more easily
    flattened_layer = tf.reshape(layer_values, [-1])

    # select the top k=5 neuron values from the flattened vector and return these values
    top_k_vals, _ = tf.math.top_k(flattened_layer, k=5)
    return top_k_vals

def get_layer_output_model():
    """
    Returns a model that will output the intermediate values of all the selected layers.
    The output of the model is a list where each element is the intermediate output of one of the selected layers.
    This output will then be fed into an MLP to classify the examples as real or fake.
    """
    # load pretrained VGG model
    vgg_model = load_pretrained_vgg_model()

    # select required layers
    layer_names = select_layers(vgg_model)

    # get the layer-wise outputs from the VGG model
    # this will create a list where each element is the output of an intermediate layer
    # each element contains all of the neuron values for the given layer, but we need to select the 5 greatest.
    hidden_layer_outputs = [vgg_model.get_layer(layer_name).output for layer_name in layer_names]

    # use the TKAN algorithm (see function above) to select the specified values from the paper
    # we are using this algorithm because it had better performance as outlined in the paper
    # each layer has a different shape, so this process has to be done manually.
    # once we have selected the 5 neuron values from each layer, we can combine all of them together.
    tkan_hidden_activations = [top_k_activated_neuron(layer_output) for layer_output in hidden_layer_outputs]

    # combine all list elements into one tensor
    combined = tf.keras.layers.concatenate(tkan_hidden_activations)

    # create and return the model
    # this model contains the VGG model plus some neuron selection algorithms
    return tf.keras.Model(inputs=vgg_model.input, outputs=combined)

def get_deepsonar_model():
    """
    Uses the tensorflow functional API to build the DeepSonar model.
    The DeepSonar model is the VGG SR system + hidden neuron selection + MLP.
    """
    # get the original model with VGG's hidden layers as output
    # don't retrain it (we'll just use the pretrained model)
    vgg_model = get_layer_output_model()
    vgg_model.trainable = False

    # DeepSonar model takes as input tensors of shape (None, 205)
    # VGG layer model outputs tensors of shape (205,)
    # reshape output from (205,) to (None, 205)
    # the None is a variable dimension and corresponds to having a variable batch size
    x = tf.expand_dims(vgg_model.output, axis=0)

    # DeepSonar model architecture
    # see section 4.4 in the paper for details
    # the dimension of each Dense layer in the MLP is not specified in the paper, so we can play around with different ones
    # trade off of bias/variance <=> underfitting and overfitting can help us choose the correct dimension
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)

    # for the final activation, it seems we can use sigmoid or softmax
    # it really makes no difference, they capture the same functionality
    # for sigmoid, I don't have to one hot encode my labels, so let's go with that
    final_output = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    # combine the hidden layer activations with the MLP (this is DeepSonar)
    deepsonar_model = tf.keras.Model(vgg_model.input, final_output)
    return deepsonar_model