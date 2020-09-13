import json
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

import load

STEP = 256

def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)['val'].squeeze()

def load_all(data_path):
    label_file = os.path.join(data_path, "REFERENCE-v3.csv")
    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]

    dataset = []
    for record, label in tqdm.tqdm(records):
        ecg_file = os.path.join(data_path, record + ".mat")
        ecg_file = os.path.abspath(ecg_file)
        ecg = load_ecg_mat(ecg_file)
        num_labels = ecg.shape[0] // STEP
        dataset.append((ecg_file, [label]*num_labels))
    return dataset

def split(dataset, dev_frac):
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    aux_train = dataset[dev_cut:]

    test = aux_train[:dev_cut]
    train = aux_train[dev_cut:]

    return train, dev, test

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            json.dump(datum, fid)
            fid.write('\n')

if __name__ == "__main__":

    # Same percentage used for testing and validation
    validation_test_percent = 0.2

    # Dataset location
    data_dir = "../ecg_datasets/PhysioNet_Challenge_Data"
    data_path_folder = os.path.abspath(data_dir)

    # Load MATLAB ECG records
    dataset = load_all(data_path_folder)

    # Split the loaded dataset into training, validation and testing
    training, validation, test = split(dataset, validation_test_percent)

    # Construct json files with ECG records and corresponding labels
    make_json("training_physionet.json", training)
    make_json("validation_physionet.json", validation)
    make_json("test_physionet.json", test)

    data_json = "../"
    data_json_path = os.path.abspath(data_json)

    # Load Datasets
    training_physionet = load.load_dataset(os.path.join(data_json_path, "training_physionet.json"))
    validation_physionet = load.load_dataset(os.path.join(data_json_path, "validation_physionet.json"))
    test_physionet = load.load_dataset(os.path.join(data_json_path, "test_physionet.json"))

    print("PhysioNet Challenge Dataset size: " + str(len(training_physionet[0]) + len(validation_physionet[0]) + len(test_physionet[0])))
    print("Training size: " + str(len(training_physionet[0])) + " records")
    print("Validation size: " + str(len(validation_physionet[0])) + " records")
    print("Testing size: " + str(len(test_physionet[0])) + " records")

