import tensorflow as tf

# just the class in this file
# the class is not used anywhere, but it was an attempt to connect VGG + DeepSonar.


class SelectNeuronLayer(tf.keras.layers.Layer):
    """
    Class that implements TKAN from DeepSonar (to connect the VGG model).
    https://www.tensorflow.org/tutorials/customization/custom_layers
    """
    def __init__(self, num_outputs):
        """
        Where you can do all input-independent initialization.
        """
        super(SelectNeuronLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        """
        Where you know the shapes of the input tensors and can do the rest of the initialization.
        """
        self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])

    def call(self, inputs):
        """
        Where you do the forward computation.
        Doesn't work here, some issue with graph mode or something like that.
        """
        return tf.matmul(inputs, self.kernel)