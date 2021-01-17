import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle


class DataGenerator(object):
    def __init__(self, batch_size):
        self.bs = batch_size

        # Constraint
        #   Vm <= 30cm/s
        #   argmax(V) - argmax(A) = 200ms
        # => Distance <= 15cm

        # One trail
        #   |--0.5s--|--1.5s--|--0.5s--|--0.2s--|
        #   |--rest--|--stim--|--rest--|--delay-|
        # Stim will random slice on the trail time window
        self.delta_time = 0.02          # 20ms
        self.stimulus_duration = 1.5    # 1.5s
        self.margin = 0.5               # 0.5s
        self.v_a_gt_delay = 0.2         # 200ms = delay
        self.v_a_gt_delay_sampling_num = int(self.v_a_gt_delay / self.delta_time)
        self.trail_duration = self.stimulus_duration + 2 * self.margin + self.v_a_gt_delay  # 2.7s
        self.trail_sampling_num = int(self.trail_duration / self.delta_time)

        # Params for velocity-time curve
        #   Velocity sd decides the time lag between the peak of velocity and accelerate
        #   when velocity-time curve follows normal distribution
        #   amplitude is applied on the bell shape curve (normal curve) of velocity
        self.max_velocity_amplitude = 15.     # 15cm
        self.min_velocity_amplitude = 0.      # 0cm
        self.max_velocity_sd = 220. / 1000.   # 220ms
        self.min_velocity_sd = 180. / 1000.   # 180ms
        # self.velocity_sd_sd = 0.02
        _mean_velocity_sd = (self.min_velocity_sd + self.max_velocity_sd) / 2.
        self.max_velocity_value = self.max_velocity_amplitude / (math.sqrt(2 * math.pi) * _mean_velocity_sd)
        self.velocity_mean = self.stimulus_duration / 2.
        self.normalization_factor = self.max_velocity_amplitude + 2 * self.max_velocity_value

        # Sampling
        self.sampling_time_point = np.arange(
            -self.velocity_mean,
            self.velocity_mean + self.delta_time,
            step=self.delta_time)
        self.sampling_point_num = len(self.sampling_time_point)

    def next_batch(self, noise_sigma=0, velocity_input_delay=0):
        assert(abs(velocity_input_delay) < 2 * self.margin)
        # velocity_input_delay - (s)
        # stimulus - [bs, sequence, features]
        #   feature[0] - Visual information, speed
        #   feature[1] - vestibular information, accelerate
        # ground truth - [bs, sequence, features]
        #   feature[0] - expected response

        # Get Raw Sequences
        v_sequences, a_sequences = self.get_raw_inputs()

        # Random Mask
        v_sequences, a_sequences = self.apply_mask(v_sequences, a_sequences)

        # Apply Delay and Add Margin
        v_sequences, a_sequences = self.add_delay_and_margin(v_sequences, a_sequences, velocity_input_delay)

        # Ground Truths
        # TODO: Use normalization or prob representation?
        gt_sequences = self.get_gts(v_sequences, a_sequences)

        # Noise
        v_sequences, a_sequences = self.add_noise(v_sequences, a_sequences, noise_sigma)

        # Inputs
        input_sequences = np.concatenate([v_sequences, a_sequences], axis=-1)

        # Directions
        input_sequences, gt_sequences = self.apply_direction(input_sequences, gt_sequences)

        return input_sequences, gt_sequences

    def get_gts(self, v_sequences, a_sequences):
        offset_v_sequences = np.concatenate([np.zeros([self.bs, self.v_a_gt_delay_sampling_num, 1]),
                                             v_sequences[:, self.v_a_gt_delay_sampling_num:, :]], axis=1)
        gt_sequences = np.cumsum(offset_v_sequences, axis=1) + np.cumsum(np.abs(a_sequences), axis=1)
        gt_sequences = gt_sequences * self.delta_time / self.normalization_factor
        return gt_sequences

    def add_delay_and_margin(self, v_sequences, a_sequences, velocity_input_delay, left_margin=None):
        # Add delay for velocity_input
        _delay_num = int(abs(velocity_input_delay) / self.delta_time)
        _zero_seq = np.zeros([self.bs, _delay_num, 1])
        if velocity_input_delay > 0:
            v_sequences = np.concatenate([_zero_seq, v_sequences], axis=1)
            a_sequences = np.concatenate([a_sequences, _zero_seq], axis=1)
        elif velocity_input_delay < 0:
            v_sequences = np.concatenate([v_sequences, _zero_seq], axis=1)
            a_sequences = np.concatenate([_zero_seq, a_sequences], axis=1)

        # Add Margin
        _empty_point_num = int((self.margin * 2 - abs(velocity_input_delay)) / self.delta_time - 1)
        if left_margin is None:
            left_margins_len = np.random.randint(low=0, high=_empty_point_num + 1, size=self.bs)
        else:
            left_margins_len = np.ones(shape=self.bs) * left_margin
            left_margins_len = left_margins_len.astype('int32')
        right_margins_len = _empty_point_num - left_margins_len

        v_sequences = list(map(self._append_zero_margins, v_sequences, left_margins_len, right_margins_len))
        a_sequences = list(map(self._append_zero_margins, a_sequences, left_margins_len, right_margins_len))
        return np.array(v_sequences), np.array(a_sequences)

    def apply_mask(self, v_sequences, a_sequences):
        masks = np.random.choice([0, 1, 2], size=self.bs)
        v_masks = np.logical_or((masks == 0), (masks == 2))[:, np.newaxis, np.newaxis]
        a_masks = np.logical_or((masks == 1), (masks == 2))[:, np.newaxis, np.newaxis]
        v_sequences = v_sequences * v_masks
        a_sequences = a_sequences * a_masks
        return v_sequences, a_sequences

    def apply_direction(self, input_sequences, gt_sequences, direction=None):
        if direction is None:
            directions = np.random.choice([-1, 1], size=(self.bs, 1, 1))
        else:
            assert direction in (-1, 1)
            directions = np.ones(shape=(self.bs, 1, 1)) * direction
        input_sequences = input_sequences * directions
        gt_sequences = gt_sequences * directions
        return input_sequences, gt_sequences

    def get_raw_inputs(self, velocity_amplitude=None):
        # Params
        if velocity_amplitude is None:
            amp_v = np.random.uniform(low=self.min_velocity_amplitude, high=self.max_velocity_amplitude, size=self.bs)
        else:
            amp_v = np.ones(shape=self.bs) * velocity_amplitude
        sd_v = np.random.uniform(low=self.min_velocity_sd, high=self.max_velocity_sd, size=self.bs)
        # Velocity and accelerate
        v_sequences = list(map(self._sampling_from_normal_distribution_pdf, np.zeros(shape=self.bs), sd_v, amp_v))
        a_sequences = list(map(self._sampling_from_derivative_of_normal_distribution_pdf, np.zeros(shape=self.bs), sd_v, amp_v))

        v_sequences = np.array(v_sequences)[:, :, np.newaxis]
        a_sequences = np.array(a_sequences)[:, :, np.newaxis]
        return v_sequences, a_sequences

    def add_noise(self, v_sequences, a_sequences, noise_sigma):
        assert noise_sigma >= 0
        if noise_sigma is not 0:
            v_sequences = v_sequences + np.random.normal(loc=0, scale=noise_sigma, size=(bs, data_generator.trail_sampling_num, 1))
            a_sequences = a_sequences + np.random.normal(loc=0, scale=noise_sigma, size=(bs, data_generator.trail_sampling_num, 1))
        return v_sequences, a_sequences

    def batch_generator(self, noise_sigma=None, velocity_input_delay=0):
        while True:
            yield self.next_batch(noise_sigma, velocity_input_delay)

    def _sampling_from_normal_distribution_pdf(self, mean=0, std=1, amplitude=1):
        distribution = norm(loc=mean, scale=std)
        sampling_values = amplitude * distribution.pdf(self.sampling_time_point)
        return sampling_values

    def _sampling_from_derivative_of_normal_distribution_pdf(self, mean=0, sd=1, amplitude=1):
        _sampling_values = self._sampling_from_normal_distribution_pdf(mean, sd, amplitude)
        sampling_values = (mean - self.sampling_time_point) / (sd * sd) * _sampling_values
        return sampling_values

    def _append_zero_margins(self, data, left_margin_len, right_margin_len):
        left_margin = np.zeros(shape=[left_margin_len, 1])
        right_margin = np.zeros(shape=[right_margin_len + self.v_a_gt_delay_sampling_num, 1])
        return np.concatenate([left_margin, data, right_margin], axis=0)

    def get_inputs_shape(self):
        return self.trail_sampling_num, 2


if __name__ == "__main__":
    # rv = norm(loc=0, scale=1)
    # print(rv.pdf(0), rv.pdf(1))
    # print(np.arange(-0.75, 0.75 + 0.02, step=0.02))

    # Visualize dataset
    bs = 16
    data_generator = DataGenerator(bs)
    inputs, gts = data_generator.batch_generator().__next__()
    print(inputs.shape, gts.shape)
    x = range(135)

    for i in range(bs):
        plt.subplot(4, 4, i+1)
        plt.plot(x, inputs[i, :, 0])
        plt.plot(x, inputs[i, :, 1])
        plt.plot(x, gts[i, :, 0] * 75)
    plt.show()

    # Generate test set
    # bs = 128
    # data_generator = DataGenerator(bs)
    # trails = None
    # for i in range(100):
    #     _inputs, _gts = data_generator.next_batch()
    #     trail = np.concatenate([_inputs, _gts], axis=-1)
    #     if trails is None:
    #         trails = trail
    #     else:
    #         trails = np.concatenate([trails, trail], axis=0)
    #
    # with open("./dataset/testset.pickle", "wb") as f:
    #     pickle.dump(trails, f)
