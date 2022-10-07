"""
Code for training DeepSonar lies in this file.
See section 4.4 (Implementation) from the paper for more details.
"""

from deepsonar_model import get_deepsonar_model

# for compiling and evaluating the model
# the following evaluation metrics are used in DeepSonar:
# Accuracy, AUC, F1, AP, FPR, FNR, EER
# we can start with accuracy and add other ones as time goes on
# https://www.tensorflow.org/api_docs/python/tf/keras/metrics
# modify list to change the different metrics for DeepSonar
# metric_list = ['accuracy']

# # params to compile() are in the DeepSonar paper
# # see section 4.4 again
# deepsonar_model.compile(
#     optimizer=tf.keras.optimizers.SGD(learning_rate=0.0003, momentum=0.9),
#     loss='binary_crossentropy', metrics=metric_list
# )

# get the model
# deepsonar = get_deepsonar_model()
# deepsonar.summary()
