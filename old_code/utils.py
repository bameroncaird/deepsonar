""" 
This file contains some random functions that may be useful throughout the project.
"""

from fast_processing import setup_multiprocessing
import numpy as np


def tensor_info(tensor, tensor_name='tensor', print_tensor=False):
    """ 
    Gives info about the rank, shape, and data type of a tensor.
    Arguments:
        tensor: the tensor to give info about
        tensor_name: the name of the tensor (used for printing output); defaults to tensor
        print_tensor: whether you want to print the full tensor; defaults to false
    """
    print("\n----------------------------------")
    print("{} info".format(tensor_name))
    print("rank: {}".format(tensor.ndim))
    print("shape: {}".format(tensor.shape))
    print("data type: {}".format(tensor.dtype))
    if print_tensor:
        print("tensor: {}".format(tensor))
    print("-------------------------------")


def get_generator_params():
    """ 
    Returns a dictionary of hyperparameters specifically for the data generator.
    """
    return {
        'dim': (257, 250, 1), 'mp_pooler': setup_multiprocessing(), 'nfft': 512,
        'spec_len': 250, 'win_length': 400, 'hop_length': 160, 'n_classes': 5994,
        'sampling_rate': 16000, 'batch_size': 16, 'shuffle': True, 'normalize': True
    }


def get_all_hyperparameters():
    """ 
    Creates the hyperparameters for the SR model.
    Returns:
        a dictionary of hyperparameters
    Some of these parameters are very old.
    """
    return {
        'batch_size': 16, 'resnet_type': 'resnet34s',  # other option: resnet34l
        'num_ghost_clusters': 2, 'num_vlad_clusters': 8, 'bottleneck_dim': 512,
        'aggregation_mode': 'gvlad',  # other options: vlad, avg
        'loss': 'softmax',  # other option: amsoftmax
        'input_shape': (257, None, 1), 'n_fft': 512, 'spectro_len': 250,
        'window_len': 400, 'hop_len': 160, 'num_classes': 5994, 'sampling_rate': 16000,
        'weights_path': '/home/cameron/hdd/experiments/voice/models/vgg_model/pretrained_vgg_weights.h5',
        'normalize': True,
        'data_path': '/media/df/wd1/deepvoice_datasets/FoR/for-norm'
    }


def top_k_values(k, np_arr):
    """ 
    Returns the top k values from an input numpy array np_arr.
    Probably pretty slow and worse than tensorflow's top_k utility.
    """
    # flatten, sort, return top k
    return np.sort(np_arr.flatten())[-k:]
