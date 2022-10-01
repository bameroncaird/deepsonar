""" 
The purpose of this file is to train the DeepSonar model.
Uses the shuffled FoR dataset that was saved to files.
"""
import numpy as np
import tensorflow as tf
from keras import models, layers
import pickle, os

from randomize_for import load_random_for
from data_prep import get_for_datalist
from ClassifierData import load_data
from utils import tensor_info


def prepare_data(mode='train'):
    """ 
    Prepares the data and the labels for a partition of the dataset.
    This is the non-shuffled FoR dataset.
    """
    data_obj = load_data(mode=mode)

    # get the data
    x_train = data_obj.get_classifier_data()

    # get the labels
    train_labels = data_obj.get_labels()
    y_train = np.asarray(train_labels).astype('float64')

    return x_train, y_train


def get_deepsonar_optimizer():
    """
    Returns a Keras optimizer as specified in the DeepSonar paper.
    optimizer: SGD, momentum 0.9, LR 0.0001, decay 1e-6
    """
    return tf.keras.optimizers.SGD(learning_rate=0.0003, momentum=0.9)


def build_deepsonar_model():
    """
    Creates the model as stated in DeepSonar.
    From the paper: 5 fully-connected layers, no other information.
    """
    # create architecture
    model = models.Sequential()
    model.add(layers.Dense(units=256, activation='relu', input_shape=(205,)))
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=1, activation='sigmoid'))

    # compile the model with custom optimizer
    optimizer = get_deepsonar_optimizer()
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model_connect():
    """
    This is probably the slowest python function I have ever written!
    This is not a function that should be used.
    The only reason I am keeping it is because it was so slow.
    """
    # get partition and labels for FoR dataset.
    partition, labels = get_for_datalist()

    # set up data generators (with mp_pooler step that I don't fully understand)
    # pooler = setup_multiprocessing()
    # train_generator = DeepSonarDataGenerator(partition['train'], labels, batch_size=64)
    # val_generator = DeepSonarDataGenerator(partition['val'], labels, batch_size=64)

    # get the model
    classifier = build_deepsonar_model()

    # train the model
    # fit_res = classifier.fit(
    #     x=train_generator, epochs=50, validation_data=val_generator
    # )
    # model.fit(x=train_generator, epochs=8, validation_data=val_generator, use_multiprocessing=True, workers=5)
    # history = model.fit(
    #     x=train_data, y=train_labels,
    #     epochs=100, batch_size=64,
    #     validation_data=(val_data, val_labels)
    # )

    # save the model
    # print(type(fit_res))
    # old saving code below (maybe use later)
    # base_save_path = "models/vgg_sr_system/debug_vgg_training"
    # model_name = "vgg_train_8_epochs"
    # full_save_path = os.path.join(base_save_path, model_name)
    # model.save(full_save_path)


def train_model():
    """ 
    The main training function for the model.
    """
    # get the data
    # train_data, train_labels = prepare_data(mode='train')
    # val_data, val_labels = prepare_data(mode='val')
    # test_data, test_labels = prepare_data(mode='test')

    # randomized data
    train_data, train_labels = load_random_for(mode='train')
    val_data, val_labels = load_random_for(mode='val')
    test_data, test_labels = load_random_for(mode='test')

    tensor_info(train_data)
    tensor_info(val_data)
    tensor_info(test_data)

    # # check if you messed up the labels
    # count_real, count_fake, total_count = 0, 0, 0
    # for label in train_labels:
    #     total_count += 1
    #     if label == Labels.REAL.value:
    #         count_real += 1
    #     else:
    #         count_fake += 1
    # for label in val_labels:
    #     total_count += 1
    #     if label == Labels.REAL.value:
    #         count_real += 1
    #     else:
    #         count_fake += 1
    # for label in test_labels:
    #     total_count += 1
    #     if label == Labels.REAL.value:
    #         count_real += 1
    #     else:
    #         count_fake += 1
    # print("Out of {} total, {} labels were real & {} labels were fake.".format(total_count, count_real, count_fake))

    # create model
    model = build_deepsonar_model()

    # fit the model and keep track of the history dict
    history = model.fit(
        x=train_data, y=train_labels,
        epochs=100, batch_size=64,
        validation_data=(val_data, val_labels)
    )

    model.evaluate(test_data, test_labels)

    # save the model
    model.save("models/DeepSonar_replicas/random_for_duplicate_jan22")

    # I don't think this dict can be pickled, so don't save it.
    # save_dict(history_dict, "results/initial.dat")

    return history.history


def save_dict(input_dict, file_path):
    """
    Uses pickle to save an input dictionary to a file path.
    """
    # check file path
    if not os.path.exists(file_path):
        print("file path does not exist")
        return

    # save dict to file
    with open(file_path, 'wb') as f:
        pickle.dump(input_dict, f)


def load_dict(file_path):
    """ 
    Loads a dictionary from a file path (in this case, the history dict from training).
    """
    # check file path
    if not os.path.exists(file_path):
        print("file path does not exist")
        return

    # load dict from file
    with open(file_path, "rb") as f:
        output_dict = pickle.load(f)
    return output_dict


def load_random_for_model():
    """ 
    Loads the saved random FoR-based model.
    """
    return models.load_model("models/DeepSonar_replicas/final_random_for_model")


# Example usage of functions to train model:
# could put all this below into another function

# h_dict = train_model()

# # plot and save to figures/
# loss_vals = h_dict['loss']
# val_loss_vals = h_dict['val_loss']
# acc_vals = h_dict['accuracy']
# val_acc_vals = h_dict['val_accuracy']

# x_axis = range(1, len(loss_vals) + 1)

# plt.plot(x_axis, loss_vals, 'ro', label='Training Loss')
# plt.plot(x_axis, val_loss_vals, 'r', label='Validation Loss')
# plt.plot(x_axis, acc_vals, 'bo', label='Training Accuracy')
# plt.plot(x_axis, val_acc_vals, 'b', label='Validation Accuracy')

# plt.title('Training and Validation Metrics')
# plt.xlabel('Epochs')
# plt.ylabel('Loss/Accuracy')
# plt.legend()

# plt.savefig('updated_figures/random_for_duplicate_jan22.png')
