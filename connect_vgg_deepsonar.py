import tensorflow as tf
from tensorflow import keras
import numpy as np

from neuron_selection import get_layer_output_model
from data_prep import get_for_datalist
from DataGenerator import DataGenerator

# help for build_connected_model():
# https://stackoverflow.com/questions/59381246/how-to-sequentially-combine-2-tensorflow-models
# https://stackoverflow.com/questions/54458202/how-to-select-top-k-elements-of-a-keras-dense-layer
# https://www.tensorflow.org/api_docs/python/tf/math/top_k
# https://www.tensorflow.org/guide/keras/functional
# https://www.tensorflow.org/tutorials/customization/custom_layers


def build_connected_model():
    """
    Uses the tensorflow functional API to connect VGG model with DeepSonar model.
    """
    # get the original model with VGG's hidden layers as output
    # don't retrain it
    vgg_model = get_layer_output_model()
    vgg_model.trainable = False

    # DeepSonar model takes as input tensors of shape (None, 205)
    # VGG layer model outputs tensors of shape (205,)
    # reshape output from (205,) to (None, 205)

    # print("current shape of output = {}".format(vgg_model.output.shape))
    # print("required shape of ds input = {}".format(deepsonar_model.input.shape))
    x = tf.expand_dims(vgg_model.output, axis=0)

    # DeepSonar model architecture
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    final_output = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    # combine both
    full_model = tf.keras.Model(vgg_model.input, final_output)

    # the following evaluation metrics are used in DeepSonar:
    # Accuracy, AUC, F1, AP, FPR, FNR, EER.
    # https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    # if you want to try more, just add to list
    metric_list = [
        'accuracy', tf.keras.metrics.Precision()
    ]

    # params to compile() are in the DeepSonar paper
    full_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0003, momentum=0.9),
        loss='binary_crossentropy', metrics=metric_list
    )
    return full_model


def train_connected_model(print_summary=False, evaluate=False):
    """
    Trains the VGG + DeepSonar model.
    """
    # get the training & validation data
    partition, labels = get_for_datalist()
    train_generator = DataGenerator(partition['train'], labels)
    val_generator = DataGenerator(partition['val'], labels)

    # get the model
    model = build_connected_model()

    if print_summary:
        model.summary()

    # training loop
    # params to fit() not specified in paper
    model.fit(
        x=train_generator, epochs=1, batch_size=64, validation_data=val_generator, 
        use_multiprocessing=True, workers=5
    )

    # evaluate if that was asked for
    if evaluate:
        test_generator = DataGenerator(partition['test'], labels)
        model.evaluate(test_generator)

    return model


def save_connected_model(model, file_path):
    """
    Saves the model to the input file path.
    """
    model.save(file_path)


def evaluate_connected_model(file_path):
    """
    Evaluates the connected model on the FoR testing split.
    """
    # load model & data, then evaluate
    model = keras.models.load_model(file_path)
    partition, labels = get_for_datalist()
    test_generator = DataGenerator(partition['test'], labels)
    model.evaluate(test_generator)


def check_frozen_layers():
    """
    I used this to verify that I had frozen the layers correctly.
    """
    # all layers should be frozen except for the last 5 dense ones
    model = build_connected_model()
    for L in model.layers:
        if len(L.trainable_weights) > 0:
            print("layer with trainable weights found: {}".format(L.name))


# save_load_path = "/home/cameron/hdd/experiments/voice/models/deepsonar"
