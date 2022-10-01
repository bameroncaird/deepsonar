""" 
The purpose of this file is to build the new training, validation, and testing splits for the 
shuffled FoR dataset.
"""
import os
import numpy as np
import tensorflow as tf
import pickle

from ClassifierData import load_data
from Labels import Labels


def load_random_for(mode='train'):
    """ 
    Loads the data from the randomized version of the FoR dataset.
    Returns the data in the form of a dictionary: dict = {'data': <data tensor>, 'labels': <label tensor>}
    Can load train, val, or test data.
    """
    # create file name
    file_name = "data/random_for/{}.dat".format(mode)
    if not os.path.exists(file_name):
        print("File path does not exist")
        return

    # load and return dictionary
    with open(file_name, "rb") as f:
        file_dict = pickle.load(f)
    return file_dict['data'], file_dict['labels']


def get_old_data():
    """ 
    Returns the old, non-shuffled FoR dataset.
    This data represents the selected hidden layer activations as tensors.
    """
    # load old objects housing the data
    old_train_obj = load_data(mode='train')
    old_val_obj = load_data(mode='val')
    old_test_obj = load_data(mode='test')

    # get data & labels from each partition
    old_train_data = old_train_obj.get_classifier_data()
    old_val_data = old_val_obj.get_classifier_data()
    old_test_data = old_test_obj.get_classifier_data()

    old_train_labels = np.asarray(old_train_obj.get_labels()).astype('float64')
    old_val_labels = np.asarray(old_val_obj.get_labels()).astype('float64')
    old_test_labels = np.asarray(old_test_obj.get_labels()).astype('float64')

    # return the data
    return (old_train_data, old_val_data, old_test_data), (old_train_labels, old_val_labels, old_test_labels)


def save_tf_tensor(file_path, tf_tensor):
    """ 
    Saves a tensorflow tensor object to a file as a numpy array.
    """
    np_arr = np.array(tf_tensor)
    np.save(file_path, np_arr)


def load_tf_tensor(file_path):
    """ 
    Loads a tensorflow tensor object from a numpy array file.
    """
    np_arr = np.load(file_path)
    return tf.convert_to_tensor(np_arr)


def generate_new_indices():
    """ 
    Randomly generates new splits for train/val/test.
    """
    # get old data
    all_old_data = get_old_data()
    data = all_old_data[0]

    # combine all the old data
    combined_data = np.concatenate((data[0], data[1], data[2]), axis=0)

    # shuffle all index numbers
    num_examples = combined_data.shape[0]
    indices = tf.range(start=0, limit=num_examples)
    shuffled_indices = tf.random.shuffle(indices)

    # save shuffled indices to a .npy file
    file_name = "data/random_for/shuffled_indices"
    save_tf_tensor(file_name, shuffled_indices)


def shuffle_data():
    """ 
    Shuffles the FoR data tensors based on random indices.
    """
    # get old data
    all_old_data = get_old_data()
    data, labels = all_old_data[0], all_old_data[1]

    # combine all the old data
    combined_data = np.concatenate((data[0], data[1], data[2]), axis=0)
    combined_labels = np.concatenate((labels[0], labels[1], labels[2]), axis=0)

    # get new indices from file
    new_indices = load_tf_tensor("data/random_for/shuffled_indices.npy")
    print(new_indices)

    # shuffle old arrays
    new_data = tf.gather(combined_data, new_indices).numpy()
    new_labels = tf.gather(combined_labels, new_indices).numpy()

    # check if you fucked up the labels
    print("Checking the labels for all data")
    count_real, count_fake, total_count = 0, 0, 0
    for lbl in new_labels:
        total_count += 1
        if lbl == Labels.REAL.value:
            count_real += 1
        else:
            count_fake += 1
    print("Out of {} total, {} labels were real & {} labels were fake.".format(total_count, count_real, count_fake))
    # tensor_info(new_data, tensor_name='New Data')
    # tensor_info(new_labels, tensor_name='New Labels')

    # verification of new data
    # uncomment this block and run to check if the new data was converted successfully
    # all_correct = True
    # for old_index_i, data_pt in enumerate(combined_data):
    #     new_index_i = new_indices[old_index_i].numpy()

    #     # check data
    #     old_data_pt = combined_data[new_index_i]
    #     new_data_pt = new_data[old_index_i]
    #     if not (old_data_pt == new_data_pt).all():
    #         all_correct = False

    #     # check label
    #     old_label = combined_labels[new_index_i]
    #     new_label = new_labels[old_index_i]
    #     if old_label != new_label:
    #         all_correct = False
    # print("All conversions correct: {}".format(all_correct))
    return new_data, new_labels


def test_tf_gather():
    """ 
    Tests the tf.gather() function.
    More for my understanding of the function than for the project.
    """
    params = np.array([
        ['m00', 'm01', 'm02'],
        ['m10', 'm11', 'm12'],
        ['m20', 'm21', 'm22']
    ])
    print("\n\nTESTING TF GATHER\n----------------------\n")
    rand_inds = tf.random.shuffle(tf.range(start=0, limit=3))
    print(rand_inds)
    print(tf.gather(params, rand_inds).numpy())
    print("\n\nDONE TESTING TF GATHER\n----------------------\n")


def build_new_splits():
    """ 
    Gets the new training, validation, and test splits from the shuffled data.
    Training data: 26941 real, 26927 fake
    Validation data: 5400 real, 5398 fake
    Testing data: 2264 real, 2370 fake

    This function is long & ugly, but it works!
    """
    # get the shuffled data
    data, labels = shuffle_data()

    # make a set with each index from 0 to 69299
    set_length = data.shape[0]
    index_set = {i for i in range(set_length)}

    # set the example length
    example_length = data.shape[1]

    ####################################

    # training split
    num_real, num_fake = 26941, 26927
    total = num_real + num_fake
    data_tensor = np.empty(shape=(total, example_length))
    label_tensor = np.empty(shape=(total))

    print("\ncreating training/real split...")
    i = 0
    for _ in range(num_real):

        # select index with real example
        random_index = np.random.randint(low=0, high=set_length)
        label = labels[random_index]
        while label != Labels.REAL.value or random_index not in index_set:
            random_index = np.random.randint(low=0, high=set_length)
            label = labels[random_index]
        data_tensor[i] = data[random_index]
        label_tensor[i] = labels[random_index]
        index_set.remove(random_index)

        if i % 1000 == 0:
            print("real training example {} / {} found: label = {}".format(i, num_real, labels[random_index]))
        i += 1

    # fake data
    print("\ncreating training/fake split...")
    for _ in range(num_fake):
        random_index = np.random.randint(low=0, high=set_length)
        label = labels[random_index]
        while label != Labels.FAKE.value or random_index not in index_set:
            random_index = np.random.randint(low=0, high=set_length)
            label = labels[random_index]
        data_tensor[i] = data[random_index]
        label_tensor[i] = labels[random_index]
        index_set.remove(random_index)

        if i % 1000 == 0:
            print("fake training example {} / {} found: label = {}".format(i, num_fake, labels[random_index]))
        i += 1

    # save training stuff to file data/random_for/train.dat
    train_dict = {
        'data': data_tensor,
        'labels': label_tensor
    }
    with open("data/random_for/train.dat", "wb") as f:
        pickle.dump(train_dict, f)

    ######################################

    # validation split
    num_real, num_fake = 5400, 5398
    total = num_real + num_fake
    data_tensor = np.empty(shape=(total, example_length))
    label_tensor = np.empty(shape=(total))

    print("\ncreating val/real split...")
    i = 0
    for _ in range(num_real):

        # select index with real example
        random_index = np.random.randint(low=0, high=set_length)
        label = labels[random_index]
        while label != Labels.REAL.value or random_index not in index_set:
            random_index = np.random.randint(low=0, high=set_length)
            label = labels[random_index]
        data_tensor[i] = data[random_index]
        label_tensor[i] = labels[random_index]
        index_set.remove(random_index)

        if i % 1000 == 0:
            print("real validation example {} / {} found: label = {}".format(i, num_real, labels[random_index]))
        i += 1

    # fake data
    print("\ncreating val/fake split...")
    for _ in range(num_fake):
        random_index = np.random.randint(low=0, high=set_length)
        label = labels[random_index]
        while label != Labels.FAKE.value or random_index not in index_set:
            random_index = np.random.randint(low=0, high=set_length)
            label = labels[random_index]
        data_tensor[i] = data[random_index]
        label_tensor[i] = labels[random_index]
        index_set.remove(random_index)

        if i % 1000 == 0:
            print("fake validation example {} / {} found: label = {}".format(i, num_fake, labels[random_index]))
        i += 1

    # save val stuff to file data/random_for/val.dat
    val_dict = {
        'data': data_tensor,
        'labels': label_tensor
    }
    with open("data/random_for/val.dat", "wb") as f:
        pickle.dump(val_dict, f)

    ######################################

    # testing split
    num_real, num_fake = 2264, 2370
    total = num_real + num_fake
    data_tensor = np.empty(shape=(total, example_length))
    label_tensor = np.empty(shape=(total))

    print("\ncreating test/real split...")
    i = 0
    for _ in range(num_real):

        # select index with real example
        random_index = np.random.randint(low=0, high=set_length)
        label = labels[random_index]
        while label != Labels.REAL.value or random_index not in index_set:
            random_index = np.random.randint(low=0, high=set_length)
            label = labels[random_index]
        data_tensor[i] = data[random_index]
        label_tensor[i] = labels[random_index]
        index_set.remove(random_index)

        if i % 1000 == 0:
            print("real test example {} / {} found: label = {}".format(i, num_real, labels[random_index]))
        i += 1

    # fake data
    print("\ncreating test/fake split...")
    for _ in range(num_fake):
        random_index = np.random.randint(low=0, high=set_length)
        label = labels[random_index]
        while label != Labels.FAKE.value or random_index not in index_set:
            random_index = np.random.randint(low=0, high=set_length)
            label = labels[random_index]
        data_tensor[i] = data[random_index]
        label_tensor[i] = labels[random_index]
        index_set.remove(random_index)

        if i % 1000 == 0:
            print("fake test example {} / {} found: label = {}".format(i, num_fake, labels[random_index]))
        i += 1

    # save test stuff to file data/random_for/test.dat
    test_dict = {
        'data': data_tensor,
        'labels': label_tensor
    }
    with open("data/random_for/test.dat", "wb") as f:
        pickle.dump(test_dict, f)
