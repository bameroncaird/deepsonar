import os
import pickle
import numpy as np

from neuron_selection import build_classifier_dataset
from utils import tensor_info


class ClassifierData:
    """
    Used for saving the shuffled FoR dataset to files.
    """

    def __init__(self, mode='train'):
        self.mode = mode
        self.data_tensor = None
        self.label_tensor = None
        self.error_paths = None

    def build_dataset(self):
        self.data_tensor, self.label_tensor, self.error_paths = build_classifier_dataset(mode=self.mode)

    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode

    def get_data(self):
        return self.data_tensor

    def get_labels(self):
        return self.label_tensor

    def get_errors(self):
        return self.error_paths

    def get_classifier_data(self):
        """ 
        The only difference between this and get_data() is that I reshaped the data to be fed into a 
            Dense layer, as this is how it works in DeepSonar.
        original shape = (samples, 41, 5), new shape = (samples, 205)
        """
        data_shape = self.data_tensor.shape
        dim0 = data_shape[0]
        dim1 = data_shape[1]
        dim2 = data_shape[2]
        new_data = np.reshape(self.data_tensor, newshape=(dim0, dim1 * dim2))
        return new_data


def save_data(mode='train'):
    """ 
    Saves shuffled FoR dataset to a file.
    Mode can be one of train, val, or test.
    """
    # build path to file
    file_name = "data/classifier/{}.dat".format(mode)
    if os.path.exists(file_name):
        print("output file exists, now building the dataset.")
        print("will save to file {}".format(file_name))
    else:
        print("output file does not exist, will not build the dataset.")
        return

    # build dataset in the object
    data_obj = ClassifierData(mode=mode)
    data_obj.build_dataset()

    # save object to file
    with open(file_name, 'wb') as f:
        pickle.dump(data_obj, f)


def load_data(mode='train', print_info=False):
    """ 
    Loads shuffled FoR from file and optionally prints info about the data.
    """
    file_name = "data/classifier/{}.dat".format(mode)
    with open(file_name, "rb") as f:
        data_obj = pickle.load(f)
    if print_info:
        tensor_info(data_obj.get_data(), tensor_name="Data")
        tensor_info(data_obj.get_labels(), tensor_name="Labels")
    return data_obj


def error_info(data_obj):
    """
    View some errors that occurred when building shuffled FoR dataset.
    """
    error_list = data_obj.get_errors()
    print("there were {} errors when building the data".format(len(error_list)))
    for i, path in enumerate(error_list):
        print("error path {}: {}".format(i + 1, path))
