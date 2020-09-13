import argparse
import json
import keras
import os
import random
import time
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger

import network
import load
import util

MAX_EPOCHS = 100


def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


# Save model each epoch
def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
                        "{val_loss:.3f}-{val_accuracy:.3f}-{epoch:03d}-{loss:.3f}-{accuracy:.3f}.hdf5")

# Save model training process
def plot_model_history(history):

    # Model accuracy

    figure_accuracy = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')

    figure_accuracy.set_size_inches(18.5, 10.5, forward=True)
    figure_accuracy.savefig('model_accuracy.png', dpi=100)

    # Model loss

    figure_loss = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    figure_loss.set_size_inches(18.5, 10.5, forward=True)
    figure_loss.savefig('model_loss.png', dpi=100)

# Model training
def train(args, params):

    print("Loading training dataset...")
    training = load.load_dataset(params['training'])

    print("Loading validation dataset...")
    validation = load.load_dataset(params['validation'])

    print("Building preprocessor...")
    preproc = load.Preproc(*training)

    print("Training dataset size: " + str(len(training[0])) + " examples.")
    print("Validation dataset size: " + str(len(validation[0])) + " examples.")

    save_dir = make_save_dir(params['save_dir'], args.experiment)

    util.save(preproc, save_dir)

    params.update({
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    model = network.build_network(**params)

    # Stop training if the loss is no longer decreasing
    # 'patience' equals the number of epoch with no improvement
    # Prevents overfitting
    stopping = keras.callbacks.EarlyStopping(patience=5)

    # Reduce the learning rate once the learning stagnates
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=params["learning_rate"] * 0.001)

    # Save full model state after each epoch
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_best_only=False)  # Boolean for saving the last best model

    csv_logger = CSVLogger('model_log.csv', append=True, separator=';')

    # Default batch size for training
    batch_size = params.get("batch_size", 32)

    # Train model using generator
    if params.get("generator", False):
        train_gen = load.data_generator(batch_size, preproc, *training)
        val_gen = load.data_generator(batch_size, preproc, *validation)
        history = model.fit(
            train_gen,
            steps_per_epoch=int(len(training[0]) / batch_size),
            epochs=MAX_EPOCHS,
            validation_data=val_gen,
            callbacks=[checkpointer, reduce_lr, stopping,csv_logger])
    else:
        # Train model using memory loaded input data
        train_x, train_y = preproc.process(*training)
        val_x, val_y = preproc.process(*validation)
        history = model.fit(
            train_x, train_y,
            validation_split=0.05,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(val_x, val_y),
            callbacks=[checkpointer, reduce_lr,stopping, csv_logger])

    plot_model_history(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name", default="default")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    train(args, params)
