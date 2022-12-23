"""
In this file, we will take care of the VGG speaker recognition (SR) model.
This is the SR model that was used in the original DeepSonar paper.
The code for this model can be found here: https://github.com/WeidiXie/VGG-Speaker-Recognition
Credit goes to the authors Xie et al. for the majority of the code in this file.
See README.md for more information about this SR model.
There are a few relevant files from this repo, but we will compartmentalize this model into a single file for the purposes of DeepSonar.
"""

# constants
WEIGHT_DECAY = 1e-4
INPUT_DIM = (499, 129, 1)

########################################################################################
########################################################################################
# define the model architecture
# as described in the paper, the architecture is a 'thin ResNet'
# this section corresponds to the file "backbone.py" in the repo linked at the top of the file
# the code in this section is copied and pasted from the original repo
from keras import layers
from keras.regularizers import l2
from keras.layers import Activation, Conv2D, Input
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D

def identity_block_2D(input_tensor, kernel_size, filters, stage, block, trainable=True):
    """The identity block is the block that has no conv layer at shortcut.
    Also called the residual block.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(WEIGHT_DECAY),
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(WEIGHT_DECAY),
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_2)(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(WEIGHT_DECAY),
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_3)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_2D(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               strides=strides,
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(WEIGHT_DECAY),
               name=conv_name_1)(input_tensor)
    #print("\n\nOUTPUT OF SECOND CONV LAYER: {}\n\n".format(x))
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(WEIGHT_DECAY),
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_2)(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=trainable,
               kernel_regularizer=l2(WEIGHT_DECAY),
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_3)(x)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='orthogonal',
                      use_bias=False,
                      trainable=trainable,
                      kernel_regularizer=l2(WEIGHT_DECAY),
                      name=conv_name_4)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, trainable=trainable, name=bn_name_4)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def resnet_2D_v1(input_dim, mode='train'):

    bn_axis = 3
    if mode == 'train':
        inputs = Input(shape=input_dim, name='input')
    else:
        inputs = Input(shape=(input_dim[0], None, input_dim[-1]), name='input')
        # inputs = Input(shape=input_dim, name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(64, (7, 7),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(WEIGHT_DECAY),
                padding='same',
                name='conv1_1/3x3_s1')(inputs)
    #print("\n\nOUTPUT OF FIRST CONV LAYER: {}\n\n".format(x1))

    x1 = BatchNormalization(axis=bn_axis, name='conv1_1/3x3_s1/bn', trainable=True)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(x1, 3, [48, 48, 96], stage=2, block='a', strides=(1, 1), trainable=True)
    x2 = identity_block_2D(x2, 3, [48, 48, 96], stage=2, block='b', trainable=True)

    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(x2, 3, [96, 96, 128], stage=3, block='a', trainable=True)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='b', trainable=True)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='c', trainable=True)
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(x3, 3, [128, 128, 256], stage=4, block='a', trainable=True)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='b', trainable=True)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='c', trainable=True)
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(x4, 3, [256, 256, 512], stage=5, block='a', trainable=True)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='b', trainable=True)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='c', trainable=True)
    y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)
    return inputs, y


def resnet_2D_v2(input_dim, mode='train'):
    bn_axis = 3
    if mode == 'train':
        inputs = Input(shape=input_dim, name='input')
    else:
        inputs = Input(shape=(input_dim[0], None, input_dim[-1]), name='input')
    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(64, (7, 7), strides=(2, 2),
                kernel_initializer='orthogonal',
                use_bias=False, trainable=True,
                kernel_regularizer=l2(WEIGHT_DECAY),
                padding='same',
                name='conv1_1/3x3_s1')(inputs)

    x1 = BatchNormalization(axis=bn_axis, name='conv1_1/3x3_s1/bn', trainable=True)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(x1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=True)
    x2 = identity_block_2D(x2, 3, [64, 64, 256], stage=2, block='b', trainable=True)
    x2 = identity_block_2D(x2, 3, [64, 64, 256], stage=2, block='c', trainable=True)
    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(x2, 3, [128, 128, 512], stage=3, block='a', trainable=True)
    x3 = identity_block_2D(x3, 3, [128, 128, 512], stage=3, block='b', trainable=True)
    x3 = identity_block_2D(x3, 3, [128, 128, 512], stage=3, block='c', trainable=True)
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(x3, 3, [256, 256, 1024], stage=4, block='a', strides=(1, 1), trainable=True)
    x4 = identity_block_2D(x4, 3, [256, 256, 1024], stage=4, block='b', trainable=True)
    x4 = identity_block_2D(x4, 3, [256, 256, 1024], stage=4, block='c', trainable=True)
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(x4, 3, [512, 512, 2048], stage=5, block='a', trainable=True)
    x5 = identity_block_2D(x5, 3, [512, 512, 2048], stage=5, block='b', trainable=True)
    x5 = identity_block_2D(x5, 3, [512, 512, 2048], stage=5, block='c', trainable=True)
    y = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)
    return inputs, y

########################################################################################
########################################################################################
# define the full model architecture, not just the backbone ResNet
# this corresponds to the thin ResNet plus the VLAD pooling layers discussed in the paper
# this section corresponds to the file "model.py" in the repo linked at the top of the file
# the code in this section is copied and pasted from the original repo
import keras
import tensorflow as tf
import keras.backend as K

class ModelMGPU(keras.Model):
    def __init__(self, ser_model, gpus, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pmodel = tf.keras.utils.multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


class VladPooling(tf.keras.layers.Layer):
    '''
    This layer follows the NetVlad, GhostVlad
    '''

    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(shape=[self.k_centers + self.g_centers, input_shape[0][-1]],
                                       name='centers',
                                       initializer='orthogonal')
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers * input_shape[0][-1])

    def call(self, x):
        # feat : bz x W x H x D, cluster_score: bz X W x H x clusters.
        feat, cluster_score = x
        num_features = feat.shape[-1]

        # softmax normalization to get soft-assignment.
        # A : bz x W x H x clusters
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(exp_cluster_score, axis=-1, keepdims=True)

        # Now, need to compute the residual, self.cluster: clusters x D
        A = K.expand_dims(A, -1)  # A : bz x W x H x clusters x 1
        feat_broadcast = K.expand_dims(feat, -2)  # feat_broadcast : bz x W x H x 1 x D
        feat_res = feat_broadcast - self.cluster  # feat_res : bz x W x H x clusters x D
        weighted_res = tf.multiply(A, feat_res)  # weighted_res : bz x W x H x clusters x D
        cluster_res = K.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_centers, :]

        cluster_l2 = K.l2_normalize(cluster_res, -1)
        outputs = K.reshape(cluster_l2, [-1, int(self.k_centers) * int(num_features)])
        return outputs


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    # return K.categorical_crossentropy(y_true, y_pred, from_logits=False)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def vggvox_resnet2d_icassp(input_dim=INPUT_DIM, num_class=8631, mode='train', args=None):
    net = args['resnet_type']
    loss = args['loss']
    vlad_clusters = args['num_vlad_clusters']
    ghost_clusters = args['num_ghost_clusters']
    bottleneck_dim = args['bottleneck_dim']
    aggregation = args['aggregation_mode']
    num_class = args['num_classes']
    # mgpu = len(keras.backend.tensorflow_backend._get_available_gpus())

    print(f"\n\nInput Dim: {input_dim}\n\n")

    if net == 'resnet34s':
        inputs, x = resnet_2D_v1(input_dim=input_dim, mode=mode)
    else:
        inputs, x = resnet_2D_v2(input_dim=input_dim, mode=mode)
    # ===============================================
    #            Fully Connected Block 1
    # ===============================================
    x_fc = tf.keras.layers.Conv2D(bottleneck_dim, (7, 1),
                                  strides=(1, 1),
                                  activation='relu',
                                  kernel_initializer='orthogonal',
                                  use_bias=True, trainable=True,
                                  kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                  bias_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                  name='x_fc')(x)

    # ===============================================
    #            Feature Aggregation
    # ===============================================
    if aggregation == 'avg':
        if mode == 'train':
            x = tf.keras.layers.AveragePooling2D((1, 5), strides=(1, 1), name='avg_pool')(x)
            x = tf.keras.layers.Reshape((-1, bottleneck_dim))(x)
        else:
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = tf.keras.layers.Reshape((1, bottleneck_dim))(x)

    elif aggregation == 'vlad':
        x_k_center = tf.keras.layers.Conv2D(vlad_clusters, (7, 1),
                                            strides=(1, 1),
                                            kernel_initializer='orthogonal',
                                            use_bias=True, trainable=True,
                                            kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                            bias_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                            name='vlad_center_assignment')(x)
        x = VladPooling(k_centers=vlad_clusters, mode='vlad', name='vlad_pool')([x_fc, x_k_center])

    elif aggregation == 'gvlad':
        x_k_center = keras.layers.Conv2D(vlad_clusters + ghost_clusters, (7, 1),
                                         strides=(1, 1),
                                         kernel_initializer='orthogonal',
                                         use_bias=True, trainable=True,
                                         kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                         bias_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
                                         name='gvlad_center_assignment')(x)
        x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')(
            [x_fc, x_k_center])

    else:
        raise IOError('==> unknown aggregation mode')

    # ===============================================
    #            Fully Connected Block 2
    # ===============================================
    x = tf.keras.layers.Dense(bottleneck_dim, activation='relu',
                              kernel_initializer='orthogonal',
                              use_bias=True, trainable=True,
                              kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                              bias_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                              name='fc6')(x)

    # ===============================================
    #            Softmax Vs AMSoftmax
    # ===============================================
    if loss == 'softmax':
        y = tf.keras.layers.Dense(num_class, activation='softmax',
                                  kernel_initializer='orthogonal',
                                  use_bias=False, trainable=True,
                                  kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                  bias_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                  name='prediction')(x)
        trnloss = 'categorical_crossentropy'

    elif loss == 'amsoftmax':
        x_l2 = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x, 1))(x)
        y = tf.keras.layers.Dense(num_class,
                                  kernel_initializer='orthogonal',
                                  use_bias=False, trainable=True,
                                  kernel_constraint=tf.keras.constraints.unit_norm(),
                                  kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                  bias_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
                                  name='prediction')(x_l2)
        trnloss = amsoftmax_loss

    else:
        raise IOError('==> unknown loss.')

    if mode == 'eval':
        y = tf.keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(x)

    # add on the binary classifier for DeepSonar

    model = keras.models.Model(inputs, y, name='vggvox_resnet2D_{}_{}'.format(loss, aggregation))

    if mode == 'train':
        # if mgpu > 1:
        #     model = ModelMGPU(model, gpus=mgpu)
        # set up optimizer.
        optimizer_name = 'adam'
        if optimizer_name == 'adam':
            opt = tf.keras.optimizers.Adam(lr=1e-3)
        elif optimizer_name == 'sgd':
            opt = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
        else:
            raise IOError('==> unknown optimizer type')
        model.compile(optimizer=opt, loss=trnloss, metrics=['acc'])
    return model

########################################################################################
########################################################################################
# in this section, we'll put the pretrained weights into the model architecture
# this is done in main.py and predict.py in the original author's code
# this is not copied and pasted from the authors because we need to do it slightly differently
# they have a command line interface with inputs, we're just defining a static function in a file
import os

def load_pretrained_vgg_model():
    """
    Loads and returns the pretrained VGG model, starting from the weights file.
    """
    # load hyperparameters
    # these are mostly default values from the original repo
    hyper_params = {
        'batch_size': 16, 'resnet_type': 'resnet34s',  # other option: resnet34l, this is how the weights were saved for the pretrained model
        'num_ghost_clusters': 2, 'num_vlad_clusters': 8, 'bottleneck_dim': 512,
        'aggregation_mode': 'gvlad',  # means ghostvlad, other options: vlad, avg
        'loss': 'softmax',  # other option: amsoftmax
        'input_shape': INPUT_DIM, 'n_fft': 512, 'spectro_len': 250,
        'window_len': 400, 'hop_len': 160, 'num_classes': 5994, 'sampling_rate': 16000,
        'weights_path': "../sr_models/vgg_sr_pretrained_weights.h5",
        'normalize': True,
        'data_path': '/media/df/wd1/deepvoice_datasets/FoR/for-norm'
    }

    # create the architecture
    # train and eval mode return different network architectures
    # it is ever so slightly different
    # in eval mode, an extra layer is tacked on to the end of the network to compute the cosine similarity
    # this is how the VGG network was originally tested using VoxCeleb1
    # in train mode, the model is compiled and the GPUs are set up for training
     # since we just need the activations and we are using the pretrained model, let's use eval mode
    # for DeepSonar, we don't need to retrain the SR system, we assume that the weights have all the necessary information
    # problem: how to properly select the time?
    # from predict.py: "the feature extraction has to be done sample by sample since each one is of different length"
    mode = 'train' # can be 'train' or 'eval', we'll do 'eval' mode since we are not training the model
    model = vggvox_resnet2d_icassp(
        input_dim=hyper_params['input_shape'], num_class=hyper_params['num_classes'],
        mode=mode, args=hyper_params
    )

    # load the weights into the architecture
    weights_path = hyper_params['weights_path']
    if not os.path.exists(weights_path):
        print("the pretrained model weights could not be loaded because the path does not exist.")
    elif not os.path.isfile(weights_path):
        print("the pretrained model weights could not be loaded because the path provided must be a file.")
    else:
        # https://github.com/WeidiXie/VGG-Speaker-Recognition/issues/46
        # the issue just shows how to properly load the model
        model.load_weights(os.path.join(weights_path), by_name=True, skip_mismatch=True)
        print("successfully loaded model & weights.")
    return model

########################################################################################
########################################################################################
