import matplotlib.pyplot as plt
import constants
from data import DataGenerator
from sklearn.decomposition import PCA
import numpy as np
import time
import os


def validate(model, bs=constants.training_batch_size):
    data_generator = DataGenerator(batch_size=bs)
    inputs, gts = data_generator.next_batch(noise_ratio=0., velocity_input_delay=0)
    preds = model.predict(inputs, bs)

    fig_dir = "./figs/" + constants.get_dir()
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    x = range(int(inputs.shape[1]))
    col_num = 4
    row_num = min(bs, 6)
    for i in range(row_num):
        plt.subplot(row_num, col_num, col_num * i + 1)
        plt.plot(x, inputs[i, :, 0])
        plt.subplot(row_num, col_num, col_num * i + 2)
        plt.plot(x, inputs[i, :, 1])
        plt.subplot(row_num, col_num, col_num * i + 3)
        plt.plot(x, gts[i, :, 0] * data_generator.normalization_factor)
        plt.subplot(row_num, col_num, col_num * i + 4)
        plt.plot(x, preds[i, :, 0] * data_generator.normalization_factor)
    plt.savefig(fig_dir + str(time.time()) + ".png")
    plt.clf()


def dynamic_system(model, bs):
    data_generator = DataGenerator(batch_size=bs)
    inputs, gts = data_generator.next_batch(noise_ratio=0., velocity_input_delay=0)
    states = model.get_rnn_states(inputs, bs)

    data_shape = data_generator.get_inputs_shape()
    seq_len = int(data_shape[1])
    features_num = int(states.shape[-1])
    samples = np.reshape(states, newshap=(bs * seq_len, features_num))
    pca = PCA(n_components=3)
    pca.fit(samples)
    print(pca.explained_variance_ratio_)

    _trajectories = pca.transform(samples)
    trajectories = np.reshape(_trajectories, newshape=(bs, seq_len, 3))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(bs):
        ax.plot(trajectories[i, :, 0], trajectories[i, :, 1], trajectories[i, :, 2])
        ax.legend()
    plt.show()


def delay_interval():
    pass


def noise_psychophysical_curve():
    pass


def neuron_type():
    pass


def spike_correlation():
    pass


def spike_causal():
    pass


def spike_sequence_diagram():
    pass


def attractor_super_plane():
    pass
