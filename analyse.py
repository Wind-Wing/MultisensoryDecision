import constants
import network
from data import DataGenerator

import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as m_patches
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.stats
import scipy.signal
from sklearn.decomposition import PCA
from functools import partial


v_noise_sigma = 0
a_noise_sigma = 0
analyse_dir = "./analyse_results/" + constants.get_dir(v_noise_sigma, a_noise_sigma)


def validate(model, v_noise_sigma=0, a_noise_sigma=0, bs=constants.training_batch_size):
    data_generator = DataGenerator(batch_size=bs)
    inputs, gts = data_generator.next_batch()
    preds = model.predict(inputs, bs)

    fig_dir = "./figs/" + constants.get_dir(v_noise_sigma, a_noise_sigma)
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
    inputs, gts = data_generator.next_batch()
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


def delay_interval(velocity_amplitude=None, direction=1):
    bs = 1
    v_delay_list = [-0.4, -0.2, 0, 0.2]
    color_list = ["blue", "red", "black", "green"]
    v_noise_sigma = 3.65 # np.sqrt(14.9)
    a_noise_sigma = 3.65

    model = build_and_load_model()
    data_generator = DataGenerator(bs)

    v_sequences_raw, a_sequences_raw = data_generator.get_raw_inputs(velocity_amplitude)

    a_modality_list = []
    v_modality_list = []
    mix_modality_list = []
    for v_delay in v_delay_list:
        if v_delay < 0:
            left_margin = (data_generator.margin + v_delay) // data_generator.delta_time
        else:
            left_margin = data_generator.margin // data_generator.delta_time
        v, a = data_generator.add_delay_and_margin(v_sequences_raw, a_sequences_raw, v_delay, left_margin=left_margin)

        v_gt = data_generator.get_gts(v, a * 0)
        a_gt = data_generator.get_gts(v * 0, a)
        mix_gt = data_generator.get_gts(v, a)

        v, a = data_generator.add_noise(v, a, v_noise_sigma, a_noise_sigma)

        _v = v * 0. * direction
        _a = a * 1. * direction
        inputs = np.concatenate([_v, _a], axis=-1)
        pred = model.predict(inputs, bs)
        a_modality_list.append([_v, _a, a_gt, pred])

        _v = v * 1. * direction
        _a = a * 0. * direction
        inputs = np.concatenate([_v, _a], axis=-1)
        pred = model.predict(inputs, bs)
        v_modality_list.append([_v, _a, v_gt, pred])

        _v = v * 1. * direction
        _a = a * 1. * direction
        inputs = np.concatenate([_v, _a], axis=-1)
        pred = model.predict(inputs, bs)
        mix_modality_list.append([_v, _a, mix_gt, pred])

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
        if direction > 0:
            plt.axvline(np.argmax(np.squeeze(res[-1][i][0])), color=color, linestyle='--')
        else:
            plt.axvline(np.argmin(np.squeeze(res[-1][i][0])), color=color, linestyle='--')

        # Super-addition validate
        plt.subplot(row_num, col_num, row_num * col_num)
        plt.plot(x, np.squeeze(res[0][i][3] + res[1][i][3]), color=color, linestyle='--')

    plt.savefig(analyse_dir + "delay_velocity" + str(time.time()) + ".png")
    plt.clf()

    # big picture
    for i in range(len(v_delay_list)):
        color = color_list[i]

        y = scipy.signal.savgol_filter(np.squeeze(res[-1][i][-1]), 31, 4)
        plt.plot(x, y, color=color)
        y = scipy.signal.savgol_filter(np.squeeze(res[0][i][3] + res[1][i][3]), 31, 4)
        plt.plot(x, y, color=color, linestyle='--')

        # plt.axvline(np.argmax(np.squeeze(abs(res[-1][i][0]))), color=color, linestyle='--')
        # if v_delay_list[i] == 0.:
        #     plt.plot(x, np.squeeze(res[-1][i][1]) / data_generator.normalization_factor, color=color, linestyle='dotted')

    plt.axhline(0, color='black')
    patches = [m_patches.Patch(color=c, label=d) for (c, d) in zip(color_list, v_delay_list)]
    plt.legend(handles=patches)

    plt.savefig(analyse_dir + constants.cell_type + "-delay_velocity-v_amp" + str(velocity_amplitude) + "direction" + str(direction) + "-" + str(time.time()) + ".png")
    plt.clf()


# Returns: True - Positive, False - Negative
def make_decisions(preds):
    avg_directions = np.mean(preds[:, :, 0], axis=1) > 0.
    vote_directions = np.mean(preds[:, :, 0] > 0, axis=1) > 0.5
    return avg_directions, vote_directions


def optimal_integration():
    bs = 1000
    model = build_and_load_model()
    func = partial(direction_discrimination_psychophysical_curve, model=model, bs=bs)
    # binary_search(func, bounds=([0, 4], [0, 4]), threshold=0.1)
    for v_noise in np.arange(0, 8, 1):
        for a_noise in np.arange(0, 15, 1):
            v_sigma, a_sigma, mix_sigma = func(v_noise, a_noise)
            print("v_noise: %f, a_noise: %f" % (v_noise, a_noise), end=", res: ")
            print(v_sigma, a_sigma, 1. / v_sigma ** 2 + 1. / a_sigma ** 2, 1. / mix_sigma ** 2)
        print()


def binary_search(func, bounds, threshold, record_dict=None):
    assert len(bounds) > 0

    def _calc_loss(args):
        if args not in record_dict:
            record_dict[args] = func(*args)
        return record_dict[args]

    if record_dict is None:
        record_dict = dict()

    args_list = None
    for bound in bounds:
        new_args_list = []
        if args_list is None:
            new_args_list = [[bound[0]], [bound[1]]]
        else:
            for _args in args_list:
                new_args_list.append(_args.append(bound[0]))
                new_args_list.append(_args.append(bound[1]))
        args_list = new_args_list
    print(args_list)

    loss_list = []
    for _args in args_list:
        loss_list.append(_calc_loss(_args))

    # Success
    return_list = []
    for i in range(len(args_list)):
        if loss_list[i] <= threshold:
            return_list.append(args_list[i])
    if len(args_list) != 0:
        return return_list

    # Fail
    if not any(loss_list * loss_list[0] < 0): return []

    # Binary search
    center_point = [(bound[0] + bound[1]) / 2. for bound in bounds]
    return_list = []
    for _args in args_list:
        new_bound = []
        for i in range(len(_args)):
            new_bound.append([_args[i], center_point[i]].sort())
        return_list.append(binary_search(func, new_bound, threshold))
    return return_list


def direction_discrimination_psychophysical_curve(v_noise, a_noise, model, bs):
    step = 1
    v_amp_list = np.arange(-15, 15 + step, step)
    data_generator = DataGenerator(bs)

    a_modality_list = []
    v_modality_list = []
    mix_modality_list = []
    for v_amp in v_amp_list:
        # print("velocity amplitude: " + str(v_amp))
        if v_amp >= 0:
            direction = 1
        else:
            direction = -1
            v_amp = abs(v_amp)

        v, a = data_generator.get_raw_inputs(velocity_amplitude=v_amp)
        # v, a = data_generator.apply_mask(v, a)
        v, a = data_generator.add_delay_and_margin(v, a, velocity_input_delay=0)
        gts = data_generator.get_gts(v, a)
        v, a = data_generator.add_noise(v, a, v_noise, a_noise)

        def get_directions(v, a):
            inputs = np.concatenate([v, a], axis=-1) * direction
            preds = model.predict(inputs, bs)
            avg_directions, vote_directions = make_decisions(preds)

            # if np.sum(a) == 0:
            #     fig_path = analyse_dir + \
            #                constants.cell_type + \
            #                "-direction_discrimination_psychophysical_curve" \
            #                "-noise" + str(noise)
            #     visualize(inputs, inputs, preds, fig_path)

            return avg_directions

        _v = v * 1
        _a = a * 0
        v_modality_list.append(np.mean(get_directions(_v, _a)))

        _v = v * 0
        _a = a * 1
        a_modality_list.append(np.mean(get_directions(_v, _a)))

        _v = v * 1
        _a = a * 1
        mix_modality_list.append(np.mean(get_directions(_v, _a)))

    plt.plot(v_amp_list, mix_modality_list, color="black")
    plt.plot(v_amp_list, v_modality_list, color="green")
    plt.plot(v_amp_list, a_modality_list, color="red")
    fig_path = analyse_dir + \
               constants.cell_type + \
               "-direction_discrimination_psychophysical_curve" \
               "-v_noise" + str(v_noise) + \
               "-a_noise" + str(a_noise) + \
               "-" + str(time.time()) + ".png"
    plt.savefig(fig_path)
    plt.clf()

    v_miu, v_sigma = fit_normal_cdf(v_amp_list, v_modality_list)
    a_miu, a_sigma = fit_normal_cdf(v_amp_list, a_modality_list)
    mix_miu, mix_sigma = fit_normal_cdf(v_amp_list, mix_modality_list)
    # print("a, v, mix \t| %f %f %f \t| %f %f %f \t| %f %f %f %f"
    #       % (a_miu, v_miu, mix_miu,
    #          a_sigma, v_sigma, mix_sigma,
    #          1. / a_sigma ** 2, 1. / v_sigma ** 2, 1. / v_sigma ** 2 + 1. / a_sigma ** 2, 1. / mix_sigma ** 2))
    return v_sigma, a_sigma, mix_sigma


def integral_model_verification():
    bs = 1
    data_generator = DataGenerator(bs)
    model = build_and_load_model()
    for level in range(-10, 11, 1):
        v = np.ones(shape=(bs, data_generator.trail_sampling_num, 1)) * level
        a = np.ones(shape=(bs, data_generator.trail_sampling_num, 1)) * level

        inputs = np.concatenate([v, a * 0], axis=-1)
        preds = model.predict(inputs, 1) * data_generator.normalization_factor
        visualize(inputs, inputs, preds, constants.cell_type + "-integral-" + "level" + str(level) + "-v")

        inputs = np.concatenate([v * 0, a], axis=-1)
        preds = model.predict(inputs, 1) * data_generator.normalization_factor
        visualize(inputs, inputs, preds, constants.cell_type + "-integral-" + "level" + str(level) + "-a")

        inputs = np.concatenate([v, a], axis=-1)
        preds = model.predict(inputs, 1) * data_generator.normalization_factor
        visualize(inputs, inputs, preds, constants.cell_type + "-integral-" + "level" + str(level) + "-mix")


def fit_normal_cdf(x_data, y_data):
    def normal_cdf(x, mu, sigma):
        return scipy.stats.norm.cdf(x, loc=mu, scale=sigma)

    plt.plot(x_data, y_data)
    popt, pcov = scipy.optimize.curve_fit(normal_cdf, x_data, y_data, bounds=([-20, 0], [15, 5]))
    plt.plot(x_data, normal_cdf(x_data, *popt))
    # plt.show()
    plt.clf()
    return popt[0], popt[1]


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


def build_and_load_model(delay=0):
    ckpt_dir = constants.get_dir(v_noise_sigma, a_noise_sigma, delay)
    model = network.RNN()
    model.load(ckpt_dir, constants.num_epochs)
    return model


if __name__ == "__main__":
    # noise_psychophysical_curve()
    # for v_amp in range(0, 16):
    #     for direction in (-1, 1):
    #         delay_interval(velocity_amplitude=v_amp, direction=direction)
    # integral_model_verification()
    optimal_integration()