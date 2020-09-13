import fnmatch
import glob
import json
import os
import random
import tqdm

import load

STEP = 256
RELABEL = {"NSR": "SINUS", "SUDDEN_BRADY": "AVB",
           "AVB_TYPE2": "AVB", "AFIB": "AF", "AFL": "AF"}


def get_all_records(path, blacklist=set()):
    records = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.ecg'):
            if patient_id(filename) not in blacklist:
                records.append(os.path.abspath(
                    os.path.join(root, filename)))
    return records


def patient_id(record):
    return os.path.basename(record).split("_")[0]


def round_to_step(n, step):
    diff = (n - 1) % step
    if diff < (step / 2):
        return n - diff
    else:
        return n + (step - diff)


def load_episodes(record, episode_extension):
    base = os.path.splitext(record)[0]
    ep_json = base + episode_extension
    ep_json = glob.glob(ep_json)[0]

    with open(ep_json, 'r') as fid:
        episodes = json.load(fid)['episodes']
    episodes = sorted(episodes, key=lambda x: x['onset'])

    for episode in episodes:
        episode['onset_round'] = round_to_step(episode['onset'], STEP)
        rn = episode['rhythm_name']
        episode['rhythm_name'] = RELABEL.get(rn, rn)

    for e, episode in enumerate(episodes):
        if e == len(episodes) - 1:
            episode['offset_round'] = episode['offset']
        else:
            episode['offset_round'] = episodes[e + 1]['onset_round'] - 1
    return episodes


def make_labels(episodes):
    labels = []
    for episode in episodes:
        rhythm_len = episode['offset_round'] - episode['onset_round'] + 1
        rhythm_labels = int(rhythm_len / STEP)
        rhythm = [episode['rhythm_name']] * rhythm_labels
        labels.extend(rhythm)
    trunc_samp = int(episodes[-1]['offset'] / STEP)
    labels = labels[:trunc_samp]
    return labels

def construct_dataset(records, episode_extension='.episodes.json'):
    data = []
    for record in tqdm.tqdm(records):
        labels = make_labels(load_episodes(record, episode_extension))
        assert len(labels) != 0, "Zero labels?"
        data.append((record, labels))
    return data

def stratify(records, validation_percent):
    pids = list(set(patient_id(record) for record in records))
    random.shuffle(pids)
    cut = int(len(pids) * validation_percent)
    validation_pids = set(pids[:cut])
    training = [r for r in records if patient_id(r) not in validation_pids]
    validation = [r for r in records if patient_id(r) in validation_pids]
    return training, validation

def load_training_validation(data_path, validation_percent, episode_extension):
    records = get_all_records(data_path)
    training, validation = stratify(records, validation_percent)
    print("Construct training dataset...")
    training = construct_dataset(training, episode_extension)
    print("Construct validation dataset...")
    validation = construct_dataset(validation, episode_extension)
    return training, validation

def load_reviewer_id(record, episode_extension):
    base = os.path.splitext(record)[0]
    ep_json = base + episode_extension
    ep_json = glob.glob(ep_json)[0]

    with open(ep_json, 'r') as fid:
        return json.load(fid)['reviewer_id']


def load_test(data_path, episode_extension):
    records = get_all_records(data_path)
    print("Constructing test...")
    test = construct_dataset(records, episode_extension)
    # Get the reviewer id
    reviewers = [load_reviewer_id(r, episode_extension) for r in records]
    test = [(e, l, r)
            for (e, l), r in zip(test, reviewers)]
    return test


def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg': d[0],
                     'labels': d[1]}
            if len(d) == 3:
                datum['reviewer'] = d[2]
            json.dump(datum, fid)
            fid.write('\n')


if __name__ == "__main__":

    data_dir = "../ecg_datasets/"
    data_path_folder = os.path.abspath(data_dir)

    data_train_validation = os.path.join(data_path_folder, "train_validation")
    data_test = os.path.join(data_path_folder, "test")
    data_rev = os.path.join(data_path_folder, "iRhythm_Data")

    # Validation dataset percentage
    validation_percent = 0.25

    # Construct datasets from cardiologists group annotations
    training, validation = load_training_validation(data_train_validation, validation_percent, '_grp*.episodes.json')
    test = load_test(data_test, '_grp*.episodes.json')

    make_json("training_irhythm.json", training)
    make_json("validation_irhythm.json", validation)
    make_json("test_irhythm.json", test)

    # Construct cardiologist annotation files
    for i in range(6):
        test = load_test(data_rev, "_rev{}.episodes.json".format(i))
        make_json("cardiologist{}.json".format(i), test)

    data_json = "../"
    data_json_path = os.path.abspath(data_json)

    training_irhythm = load.load_dataset(os.path.join(data_json, "training_irhythm.json"))
    validation_irhythm = load.load_dataset(os.path.join(data_json, "validation_irhythm.json"))
    test_irhythm = load.load_dataset(os.path.join(data_json, "test_irhythm.json"))

    print("iRhythm Dataset size: " + str(len(training_irhythm[0]) + len(validation_irhythm[0]) + len(test_irhythm[0])))
    print("Training size: " + str(len(training_irhythm[0])) + " records")
    print("Validation size: " + str(len(validation_irhythm[0])) + " records")
    print("Testing size: " + str(len(test_irhythm[0])) + " records")


