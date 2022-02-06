import os

from sr_model import vggvox_resnet2d_icassp
from utils import get_all_hyperparameters

# provides functions for loading the pretrained VGG model.


def load_pretrained_model():
    """ 
    Loads the pretrained VGG model weights.
    Currently used for training connected VGG and DeepSonar model.
    """
    # load hyperparameters
    params = get_all_hyperparameters()

    # create the architecture
    # training and eval mode return different network architectures
    # works because you save the weights to a file after training model
    # might be something to consider, but for now just do eval mode
    model = vggvox_resnet2d_icassp(
        input_dim=params['input_shape'], num_class=params['num_classes'],
        mode='eval', args=params
    )

    # load the weights into the architecture
    weights_path = params['weights_path']
    if not os.path.exists(weights_path):
        print("the pretrained model weights could not be loaded because the path does not exist.")
    elif not os.path.isfile(weights_path):
        print("the pretrained model weights could not be loaded because the path provided must be a file.")
    else:
        # https://github.com/WeidiXie/VGG-Speaker-Recognition/issues/46
        model.load_weights(os.path.join(weights_path), by_name=True, skip_mismatch=True)
        print("successfully loaded model & weights.")
    return model


def load_pretrained_model_from_weights(file_path="models/vgg_sr_system/sr_model_weights.h5", mode='train'):
    """
    Loads a pretrained model from a weights file path.
    This function is slightly different from the one above, with better parameters.
    Currently used for training the VGG model.
    """
    if not os.path.exists(file_path):
        print("Could not load pretrained model as file path does not exist.")
        return

    # set parameters for pretrained model
    # these are default params from the open source project
    model_params = {
        'input_dim': (257, None, 1), 'num_classes': 5994, 'resnet_type': 'resnet34s',
        'loss': 'softmax', 'num_vlad_clusters': 8, 'num_ghost_clusters': 2,
        'bottleneck_dim': 512, 'aggregation_mode': 'gvlad'
    }

    # create model architecture (slightly different for training & evaluation).
    model = vggvox_resnet2d_icassp(
        input_dim=model_params['input_dim'], num_class=model_params['num_classes'],
        mode=mode, args=model_params
    )

    # add the weights
    model.load_weights(os.path.join(file_path), by_name=True, skip_mismatch=True)
    return model

