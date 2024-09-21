import mne
import numpy as np
import torch
import torch.nn as nn
import os
import scipy.io as sio
import matplotlib

matplotlib.use('Qt5Agg')




def pre(root_x, root_y, name_x, name_y, exist_question=False, test=False):
    filename_x = os.path.join(root_x, name_x)
    filename_y = os.path.join(root_y, name_y)

    raw = mne.io.read_raw_gdf(filename_x)

    events, events_dict = mne.events_from_annotations(raw)
    print(events)
    print(events_dict)
    raw.load_data()

    # raw.filter(4., 40., fir_design='firwin')

    raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                           exclude='bads')

    tmin, tmax = 0., 4.
    event_id = dict({'769': 10, '770': 11})
    if exist_question:
        event_id = dict({'769': 4, '770': 5})
    if test:
        event_id = dict({'783': 11})
    if exist_question==True and test==True:
        event_id = dict({'783': 5})
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True)
    x_data = epochs.get_data() * 1e6
    x_data = x_data[:, :, :1000]

    # y_label=epochs.events[:, -1]- min(epochs.events[:, -1])
    y_label = sio.loadmat(filename_y)["classlabel"]
    # epochs.plot()
    return x_data, y_label

filename = r"E:\notebook\pyproject\EEG-Motor-Imagery-Classification---ANN-master\BAC_2B\BCICIV_2b_gdf\B0101T.gdf"
# root_x = r"E:\notebook\pyproject\EEG-Motor-Imagery-Classification---ANN-master\BAC_2B\BCICIV_2b_gdf"
root_x = r"D:\download\MT-MBCNN-main\dataB"
# root_y = r"E:\notebook\pyproject\EEG-Motor-Imagery-Classification---ANN-master\BAC_2B\true_labels"
root_y = r"D:\download\MT-MBCNN-main\dataB"
# x,y=pre(root_x,root_y,"B0205E.gdf","B0205E.mat",test=True)
# print(y.shape)
for w in range(1,10):
    for i in range(1, 6):

        name = f"B0{w}0{i}"
        if i < 4:
            name_x = name + str("T.gdf")
            name_y = name + str("T.mat")
            test = False
        else:
            name_x = name + str("E.gdf")
            name_y = name + str("E.mat")
            test = True
        exist_question=False

        if w==1:
            if i == 2:
                exist_question = True
            else:
                exist_question = False
        if w==5:
            if i == 4:
                exist_question = True
            else:
                exist_question = False
        # print(name)
        x, y = pre(root_x, root_y, name_x, name_y, test=test, exist_question=exist_question)

        if i == 1:
            x_train = x
            y_train = y
        if (1 < i) and (i < 4):
            x_train = np.concatenate((x_train, x), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)
        if i == 4:
            x_test = x
            y_test = y
        if (4 < i) and (i < 6):
            x_test = np.concatenate((x_test, x), axis=0)
            y_test = np.concatenate((y_test, y), axis=0)

    x_train = x_train[:, :, :1000]
    x_test = x_test[:, :, :1000]

    from data_utils import *

    # filtBank = [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]]
    filtBank = [[4, 7], [8, 13], [14, 30], [31, 50]]
    transform = filterBank(filtBank, 250)
    # numfiltBank = 9
    numfiltBank = 4

    print(f'Processing subject No. {w}')
    train_data = x_train
    print(train_data.shape)
    train_label = y_train
    test_data = x_test
    test_label = y_test

    print(2222222222222222)
    print(train_data.shape)
    print(test_data.shape)

    multifreq_train_data = np.zeros([train_data.shape[0], numfiltBank, *train_data.shape[1:3]])
    multifreq_test_data = np.zeros([test_data.shape[0], numfiltBank, *test_data.shape[1:3]])
    for j in range(train_data.shape[0]):
        multifreq_train_data[j, :, :, :] = transform(train_data[j])
    for j in range(test_data.shape[0]):
        multifreq_test_data[j, :, :, :] = transform(test_data[j])
    print("11111111111111")
    print(multifreq_train_data.shape)

    import scipy.io
    data = {'data': multifreq_train_data, 'label': y_train.reshape(-1, 1)}
    scipy.io.savemat(fr'E:\project\EEG-IMCTNet\Data\2b_four\A0{str(w)}T.mat', data)
    data = {'data': multifreq_test_data, 'label': y_test.reshape(-1, 1)}
    scipy.io.savemat(fr'E:\project\EEG-IMCTNet\Data\2b_four\A0{str(w)}E.mat', data)