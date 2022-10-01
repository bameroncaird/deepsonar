import numpy as np
import tensorflow as tf
import data_prep as dp

from Labels import Labels


def view_final_vgg_input_shape():
    """
    Loads an audio file and reshapes it exactly how it's done in DataGenerator.
    This represents the final input shape of input tensors to VGG.
    """
    # hard coded file path to example
    filepath = "/home/cameron/voice_data/fake-or-real/for-norm/training/fake/file20440.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav"
    
    # exact line of code (line 104) from DataGenerator (VGG authors implemented this)
    d = np.expand_dims(np.expand_dims(dp.load_data(filepath), 0), -1)
    print("\nvgg input data shape = {}\n".format(d.shape))


class DataGenerator(tf.keras.utils.Sequence):
    """
    Implemented by authors of VGG, but I made some changes to their code.
    Much of code comes from here: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    A data generator yields data batch-by-batch instead of loading it all at once.
    It's an old solution, but it utilizes the GPU well.
    """

    def __init__(self, list_IDs, labels, mp_pooler=None, dim=(257, 250, 1), augmentation=True, batch_size=64, nfft=512, spec_len=250,
                 win_length=400, sampling_rate=16000, hop_length=160, n_classes=5994, shuffle=True, normalize=True):
        """ 
        Initializer for the data generator.
        """
        self.indexes = None
        self.dim = dim
        self.nfft = nfft
        self.sr = sampling_rate
        self.spec_len = spec_len
        self.normalize = normalize
        self.mp_pooler = mp_pooler
        self.win_length = win_length
        self.hop_length = hop_length

        self.labels = labels
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X, y = self.__data_generation_mp(list_IDs_temp, indexes)
        # try replacing above line with following line if above line doesn't work (or vice versa)
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        The main function that makes this class work.
        Generates data containing batch_size samples.
        """
        # keep track of number of real & fake examples in the batch
        # keep track of errors in loading the data
        # these variables only exist while the batch is being processed
        num_real = 0; num_fake = 0; num_except = 0

        # initialize tensors
        data = np.empty(shape=(self.batch_size, *self.dim))
        labels = np.empty(shape=(self.batch_size), dtype=int)

        # generate data
        for i, file_path in enumerate(list_IDs_temp):
            
            # there are just a few paths that are incorrect
            # current solution is to add the previous data and label
            try:
                # store data and label
                # shape of load_data is (257, 250)
                # input shape for training data: (257, 250, 1)
                # the two calls to expand_dims() are from the authors of the VGG model.
                # the two calls to expand_dims() reshape the data correctly.
                # data[i].shape = (1, 257, 250, 1)
                # view_final_vgg_input_shape() shows this to you.
                data[i] = np.expand_dims(np.expand_dims(dp.load_data(file_path), 0), -1)
                labels[i] = self.labels[file_path]

            except:
                # just store the previous one
                num_except += 1
                print("exception {} in batch, storing duplicate data".format(num_except))
                data[i] = data[i - 1]
                labels[i] = labels[i - 1]

            if labels[i] == Labels.FAKE.value:
                num_fake += 1
            else:
                num_real += 1

        # check class balance in the batch
        # print("\nnum fake = {}, num real = {}\n".format(num_fake, num_real))

        # categorically encode labels & return
        return data, tf.keras.utils.to_categorical(labels, num_classes=self.n_classes)

    def __data_generation_mp(self, list_IDs_temp, indexes):
        """ 
        A modified version of __data_generation() implemented by the authors of VGG.
        I currently do not use this function.
        """
        X = [self.mp_pooler.apply_async(dp.load_data,
                                        args=(ID, self.win_length, self.sr, self.hop_length,
                                              self.nfft, self.spec_len)) for ID in list_IDs_temp]
        X = np.expand_dims(np.array([p.get() for p in X]), -1)

        y = np.empty(shape=(self.batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            y[i] = self.labels[ID]

        # this line does not work for our code, but was from the open-source implementation.
        # y = self.labels[indexes]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
