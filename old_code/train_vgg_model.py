"""
The purpose of this file is to train the VGG model.
I never actually got this to train well.
"""
import os

import data_prep as dp
from DataGenerator import DataGenerator
from load_sr_model import load_pretrained_model_from_weights
from fast_processing import setup_multiprocessing


def train_model(use_pretrained=True):
    """
    The main function for training the model.
    If use_pretrained is True, the pretrained model is used for initializing the weights.
    Otherwise, the weights are initialized how they usually are, not using the pretrained weights.
    """
    # get partition & labels for data generator
    partition, labels = dp.get_vgg_data_dicts()

    # set up data generators (with mp_pooler step that I don't fully understand)
    pooler = setup_multiprocessing()
    train_generator = DataGenerator(partition['train'], labels, mp_pooler=pooler)
    val_generator = DataGenerator(partition['val'], labels, mp_pooler=pooler)

    # get the model
    if use_pretrained:
        # mode will always be 'train' in this file since we are training the model.
        model = load_pretrained_model_from_weights(mode='train')
    else:
        print("not doing this right now. returning and not training...")
        return

    # train the model
    model.fit(x=train_generator, epochs=8, validation_data=val_generator, use_multiprocessing=True, workers=5)

    # save the model
    # for now, we are going to test various different training configurations, then evaluate these models.
    base_save_path = "models/vgg_sr_system/debug_vgg_training"
    model_name = "vgg_train_8_epochs"
    full_save_path = os.path.join(base_save_path, model_name)
    model.save(full_save_path)
