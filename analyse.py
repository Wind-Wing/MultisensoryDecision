import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import constants
from data import DataGenerator
from sklearn.decomposition import PCA
import numpy as np
import time
import os
import network
import data


analyse_dir = "./analyse_results/"


def validate(model, bs=constants.training_batch_size):
    data_generator = DataGenerator(batch_size=bs)
    inputs, gts = data_generator.next_batch(noise_ratio=0., velocity_input_delay=0)
    preds = model.predict(inputs, bs)

    fig_dir = "./figs/" + constants.get_dir()
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    visualize(inputs, gts, preds, fig_dir)


def visualize(inputs, gts, preds, prefix):
    x = range(int(inputs.shape[1]))
    bs = int(inputs.shape[0])
    col_num = 4
    row_num = min(bs, 6)
    for i in range(row_num):
        plt.subplot(row_num, col_num, col_num * i + 1)
        plt.plot(x, inputs[i, :, 0])
        plt.subplot(row_num, col_num, col_num * i + 2)
        plt.plot(x, inputs[i, :, 1])
        plt.subplot(row_num, col_num, col_num * i + 3)
        plt.plot(x, gts[i, :, 0])
        plt.subplot(row_num, col_num, col_num * i + 4)
        plt.plot(x, preds[i, :, 0])
    plt.savefig(prefix + str(time.time()) + ".png")
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
    bs = 1
    v_delay_list = [-0.2, 0, 0.2]
    color_list = ["red", "black", "green"]

    model = build_and_load_model(noise_ratio=0.1)
    data_generator = data.DataGenerator(bs)

    v_sequences_raw, a_sequences_raw = data_generator.get_raw_inputs()

    a_modality_list = []
    v_modality_list = []
    mix_modality_list = []
    for v_delay in v_delay_list:
        if v_delay < 0:
            left_margin = (data_generator.margin + v_delay) // data_generator.delta_time
        else:
            left_margin = data_generator.margin // data_generator.delta_time
        v, a = data_generator.add_delay_and_margin(v_sequences_raw, a_sequences_raw, v_delay, left_margin=left_margin)

        _v = v * 0.
        _a = a * 1.
        gt = data_generator.get_gts(_v, _a)
        inputs = np.concatenate([_v, _a], axis=-1)
        pred = model.predict(inputs, bs)
        a_modality_list.append([_v, _a, gt, pred])

        _v = v * 1.
        _a = a * 0.
        gt = data_generator.get_gts(_v, _a)
        inputs = np.concatenate([_v, _a], axis=-1)
        pred = model.predict(inputs, bs)
        v_modality_list.append([_v, _a, gt, pred])

        _v = v * 1.
        _a = a * 1.
        gt = data_generator.get_gts(_v, _a)
        inputs = np.concatenate([_v, _a], axis=-1)
        pred = model.predict(inputs, bs)
        mix_modality_list.append([_v, _a, gt, pred])

    x = range(int(data_generator.get_inputs_shape()[0]))
    row_num = 3
    col_num = 4
    res = [v_modality_list, a_modality_list, mix_modality_list]

    for i in range(len(v_delay_list)):
        color = color_list[i]
        for j in range(len(res)):
            for k in range(col_num):
                plt.subplot(row_num, col_num, 1 + k + col_num * j)
                plt.plot(x, np.squeeze(res[j][i][k]), color=color)
        plt.axvline(np.argmax(np.squeeze(res[-1][i][0])), color=color, linestyle='--')

        # Super-addition validate
        plt.subplot(row_num, col_num, row_num * col_num)
        plt.plot(x, np.squeeze(res[0][i][3] + res[1][i][3]), color=color, linestyle='--')

    plt.savefig(analyse_dir + "delay_velocity-" + str(time.time()) + ".png")
    plt.clf()

    # big picture
    for i in range(len(v_delay_list)):
        color = color_list[i]
        plt.plot(x, np.squeeze(res[-1][i][-1]), color=color)

        plt.axvline(np.argmax(np.squeeze(res[-1][i][0])), color=color, linestyle='--')
        plt.plot(x, np.squeeze(res[0][i][3] + res[1][i][3]), color=color, linestyle='--')

        if v_delay_list[i] == 0.:
            plt.plot(x, np.squeeze(res[-1][i][1]) / data_generator.normalization_factor, color=color, linestyle='dotted')

    plt.axhline(0, color='black')
    patches = [mpatches.Patch(color=c, label=d) for (c, d) in zip(color_list, v_delay_list)]
    plt.legend(handles=patches)

    plt.savefig(analyse_dir + "delay_velocity-" + str(time.time()) + ".png")
    plt.clf()


def noise_psychophysical_curve():
    bs = 10000
    noise_ratio_list = np.arange(0, 5, 0.1)

    model = build_and_load_model(noise_ratio=0.1)
    data_generator = data.DataGenerator(bs)
    avg_res = []
    vote_res = []
    for noise_ratio in noise_ratio_list:
        print("Noise Level " + str(noise_ratio))
        inputs, gts = data_generator.next_batch(noise_ratio=noise_ratio)
        preds = model.predict(inputs, bs)
        # visualize(inputs, gts, preds, analyse_dir + str(noise_ratio) + "-")

        gt_direction = gts[:, -1, 0] > 0
        avg_direction = np.mean(preds[:, :, 0], axis=1) > 0.
        vote_direction = np.mean(preds[:, :, 0] > 0, axis=1) > 0.5
        avg_direction_acc = np.mean(np.equal(gt_direction, avg_direction))
        vote_direction_acc = np.mean(np.equal(gt_direction, vote_direction))
        avg_res.append(avg_direction_acc)
        vote_res.append(vote_direction_acc)

    plt.plot(noise_ratio_list, avg_res, color="red")
    plt.plot(noise_ratio_list, vote_res, color="green")
    plt.savefig(analyse_dir + "noise_psychophysical_curve-" + str(time.time()) + ".png")
    plt.clf()


def neuron_type():
    # TODO: response to what signal
    pass


def spike_correlation():
    pass


def spike_causal():
    pass


def spike_sequence_diagram():
    pass


def attractor_super_plane():
    pass


def build_and_load_model(noise_ratio=None, delay=0):
    ckpt_dir = constants.get_dir(noise_ratio, delay)
    model = network.RNN()
    model.load(ckpt_dir, constants.num_epochs)
    return model


if __name__ == "__main__":
    # noise_psychophysical_curve()
    delay_interval()
