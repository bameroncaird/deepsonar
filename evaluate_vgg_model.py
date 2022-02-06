import numpy as np
import os
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

import data_prep as dp


def calculate_eer(y, y_score):
    """
    Implemented by VGG model authors.
    """
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def evaluate_model(model, split_name="veri_test.txt"):
    """
    Evaluates input VGG model on an input VoxCeleb1 test split.
    Currently only uses the EER (Equal Error Rate) metric.
    Most of this code comes from VGG model's GitHub.
    There are six different .txt files to evaluate on. Here are their names:
    - veri_test.txt
    - veri_test2.txt
    - list_test_hard.txt
    - list_test_hard2.txt
    - list_test_all.txt
    - list_test_all2.txt
    Each test split is slightly different; for example, list_test_hard.txt tests using pairs
        with similar genders.
    More info found here: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
    """
    # build paths to verification list and raw data
    # replace the file name at the end of the testing path with the desired test split.
    test_list_path = "/home/cameron/voice_data/voxceleb1/test_lists/{}".format(split_name)
    base_data_path = "/home/cameron/voice_data/voxceleb1/wav"
    if not os.path.exists(test_list_path):
        print("Error: Path to evaluation file does not exist, can't evaluate...")
        return

    # structure for each line of verify_list: <label> <id1 clip> <id2 clip>
    # if id1 == id2, label == 1. else, label == 0.
    # the rest of the function is completely copied from the author's code:
    verify_list = np.loadtxt(test_list_path, str)
    verify_labels = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(base_data_path, i[1]) for i in verify_list])
    list2 = np.array([os.path.join(base_data_path, i[2]) for i in verify_list])

    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list)

    # The feature extraction process has to be done sample-by-sample
    #   because each sample is of different lengths.
    total_length = len(unique_list)
    feats, scores, labels = [], [], []
    for c, ID in enumerate(unique_list):
        if c % 50 == 0:
            print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
        specs = dp.load_data(path=ID, mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)

        v = model.predict(specs)
        feats += [v]

    feats = np.array(feats)

    # ==> compute the pair-wise similarity.
    for c, (p1, p2) in enumerate(zip(list1, list2)):
        ind1 = np.where(unique_list == p1)[0][0]
        ind2 = np.where(unique_list == p2)[0][0]

        v1 = feats[ind1, 0]
        v2 = feats[ind2, 0]

        scores += [np.sum(v1 * v2)]
        labels += [verify_labels[c]]
        print('scores : {}, gt : {}'.format(scores[-1], verify_labels[c]))

    scores = np.array(scores)
    labels = np.array(labels)

    # uncomment if you want to save the results somewhere in the file system
    # np.save(os.path.join(result_path, 'prediction_scores.npy'), scores)
    # np.save(os.path.join(result_path, 'groundtruth_labels.npy'), labels)

    eer, _ = calculate_eer(labels, scores)
    print('==> model EER: {}'.format(eer))
