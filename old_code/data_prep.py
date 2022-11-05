import librosa
import numpy as np
import os
import pickle

# from Labels import Labels
import utils as ut

# file provides various data-related functions


# def get_for_datalist(**params):
#     """
#     Returns partition and labels dictionaries.
#     partition = { 'train': <paths>, 'val': <paths>, 'test': <paths> }
#     labels = { 'path': label for all paths in all splits }
#     Used for DataGenerator.
#     """
#     partition, labels = {}, {}
#     data_path = '/home/cameron/voice_data/fake-or-real/for-norm'

#     # training partition
#     training_list = []
#     training_path_real = os.path.join(data_path, 'training', 'real')
#     training_path_fake = os.path.join(data_path, 'training', 'fake')
#     for file_name in os.listdir(path=training_path_real):
#         audio_path = os.path.join(training_path_real, file_name)
#         training_list.append(audio_path)
#         labels[audio_path] = Labels.REAL.value
#     for file_name in os.listdir(path=training_path_fake):
#         audio_path = os.path.join(training_path_fake, file_name)
#         training_list.append(audio_path)
#         labels[audio_path] = Labels.FAKE.value
#     partition['train'] = training_list

#     # validation partition
#     val_list = []
#     val_path_real = os.path.join(data_path, 'validation', 'real')
#     val_path_fake = os.path.join(data_path, 'validation', 'fake')
#     for file_name in os.listdir(path=val_path_real):
#         audio_path = os.path.join(val_path_real, file_name)
#         val_list.append(audio_path)
#         labels[audio_path] = Labels.REAL.value
#     for file_name in os.listdir(path=val_path_fake):
#         audio_path = os.path.join(val_path_fake, file_name)
#         val_list.append(audio_path)
#         labels[audio_path] = Labels.FAKE.value
#     partition['val'] = val_list

#     # testing partition
#     test_list = []
#     test_path_real = os.path.join(data_path, 'testing', 'real')
#     test_path_fake = os.path.join(data_path, 'testing', 'fake')
#     for file_name in os.listdir(path=test_path_real):
#         audio_path = os.path.join(test_path_real, file_name)
#         test_list.append(audio_path)
#         labels[audio_path] = Labels.REAL.value
#     for file_name in os.listdir(path=test_path_fake):
#         audio_path = os.path.join(test_path_fake, file_name)
#         test_list.append(audio_path)
#         labels[audio_path] = Labels.FAKE.value
#     partition['test'] = test_list

#     return partition, labels


def get_vgg_data_dicts():
    """
    Returns the data partition & label lists for the VGG model (VoxCeleb2 dataset).
    Used for DataGenerator.
    """
    # partition & labels are saved to files this time
    file_name = "/home/cameron/voice_data/voxceleb2/dev/meta/partition.pkl"
    with open(file_name, "rb") as pf:
        partition = pickle.load(pf)

    file_name = "/home/cameron/voice_data/voxceleb2/dev/meta/labels.pkl"
    with open(file_name, "rb") as pf:
        labels = pickle.load(pf)

    return partition, labels


def load_wav(file_path, sample_rate):
    """ 
    Loads a .wav file into a tensor.
    """
    wav, sr = librosa.load(path=file_path, sr=sample_rate)
    if sample_rate != sr:
        print("there was an error loading your .wav file.")
    return wav


def load_wav_augment(file_path, sample_rate, mode='train'):
    """ 
    The same thing as load_wav(), but with some data augmentation.
    This is the version used in VGG repository.
    """
    wav, sr = librosa.load(file_path, sr=sample_rate)

    if sample_rate != sr:
        print("there was an error loading your .wav file.")

    # data augmentation
    if mode == 'train':
        # double the length of the wav
        extended_wav = np.append(wav, wav)
        # 30% of the time
        if np.random.random() < 0.3:
            # reverse the wav
            extended_wav = extended_wav[::-1]
    else:
        # double length of wav and reverse second half
        extended_wav = np.append(wav, wav[::-1])
    return extended_wav


def load_spectrogram(file_path, sample_rate, hop_len, window_len, n_fft=1024, mode='train'):
    """ 
    Loads a spectrogram from a .wav file.
    This is VGG's implementation.
    """
    # replace with load_wav_augment if necessary (if it's too slow)
    # wav = load_wav(file_path, sample_rate)
    wav = load_wav_augment(file_path, sample_rate, mode)
    stft = librosa.stft(y=wav, n_fft=n_fft, win_length=window_len, hop_length=hop_len)
    return stft.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    """ 
    Loads the input data for the VGG model.
    This function returns a spectrogram of shape (257, 250).
    This was implemented by the authors of the VGG model.
    """
    linear_spect = load_spectrogram(path, sr, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    _, time = mag_T.shape
    if mode == 'train':
        if time > spec_len:
            randtime = np.random.randint(0, time - spec_len)
            spec_mag = mag_T[:, randtime:randtime + spec_len]
        else:
            spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)
