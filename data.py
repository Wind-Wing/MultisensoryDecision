import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class DataGenerator(object):
    def __init__(self, batch_size):
        self.bs = batch_size

        # Constraint
        #   Vm <= 30cm/s
        #   argmax(V) - argmax(A) = 200ms
        # => Distance <= 15cm

        # One trail
        #   |--0.5s--|--1.5s--|--0.5s--|
        #   |--rest--|--stim--|--rest---|
        # Stim will random slice on the trail time window
        self.delta_time = 20. / 1000.   # 20ms
        self.stimulus_duration = 2.5    # 2.5s
        self.margin = 0.5               # 0.5s
        self.trail_duration = self.stimulus_duration + 2 * self.margin       # 2.5s
        self.trail_sampling_point_num = int(self.trail_duration / self.delta_time)

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

        # Params for noise
        # self.v_a_max_ratio = self.mean_velocity_sd * math.sqrt(math.e)
        # self.max_velocity_noise_value = self.max_velocity_value / 10.
        self.max_noise_ratio = 1.
        self.min_noise_ratio = 0.

        # Sampling
        self.sampling_time_point = np.arange(
            -self.velocity_mean,
            self.velocity_mean + self.delta_time,
            step=self.delta_time)
        self.sampling_point_num = len(self.sampling_time_point)
        self.empty_point_num = self.trail_sampling_point_num - self.sampling_point_num

    def next_batch(self):
        # stimulus - [bs, sequence, features]
        # feature[0] - Visual information, speed
        # feature[1] - vestibular information, accelerate
        # ground truth - [bs, sequence, features]
        # feature[0] - expected response
        # TODO: save the training samples. Or just create a static data set
        # TODO: do I need give time for the RNN to process

        # TODO: Add the delay in the gt

        # Params
        sd_v = np.random.uniform(low=self.min_velocity_sd, high=self.max_velocity_sd, size=self.bs)
        amp_v = np.random.uniform(low=self.min_velocity_amplitude, high=self.max_velocity_amplitude, size=self.bs)

        # Velocity and accelerate
        v_sequences = list(map(self._sampling_from_normal_distribution_pdf, np.zeros(shape=self.bs), sd_v, amp_v))
        a_sequences = list(map(self._sampling_from_derivative_of_normal_distribution_pdf, np.zeros(shape=self.bs), sd_v, amp_v))

        # Random Mask
        masks = np.random.choice([0, 1, 2], size=self.bs)
        v_masks = np.logical_or((masks == 0), (masks == 2))[:, np.newaxis]
        a_masks = np.logical_or((masks == 1), (masks == 2))[:, np.newaxis]
        v_sequences = v_sequences * v_masks
        a_sequences = a_sequences * a_masks

        # Add Margin
        left_margins_len = np.random.randint(low=0, high=self.empty_point_num+1, size=self.bs)
        right_margins_len = self.empty_point_num - left_margins_len
        v_sequences = list(map(self._append_zero_margins, v_sequences, left_margins_len, right_margins_len))
        a_sequences = list(map(self._append_zero_margins, a_sequences, left_margins_len, right_margins_len))

        # Ground Truths
        # TODO: Does gt need to be the results of discretization add up or integral of continued function?
        # TODO: Use normalization or prob representation?
        v_sequences = np.array(v_sequences)[:, :, np.newaxis]
        a_sequences = np.array(a_sequences)[:, :, np.newaxis]
        gt_sequences = np.cumsum(v_sequences, axis=1) + np.cumsum(np.abs(a_sequences), axis=1)
        gt_sequences = gt_sequences * self.delta_time / (self.max_velocity_amplitude + 2 * self.max_velocity_value)

        # Noise
        noise_ratio = np.random.uniform(low=self.min_noise_ratio, high=self.max_noise_ratio, size=(self.bs, 1))
        v_sequences = self._add_noise(v_sequences, noise_ratio)
        a_sequences = self._add_noise(a_sequences, noise_ratio)

        # Inputs
        input_sequences = np.concatenate([v_sequences, a_sequences], axis=-1)

        # Directions
        directions = np.random.choice([-1, 1], size=(self.bs, 1, 1))
        input_sequences = input_sequences * directions
        gt_sequences = gt_sequences * directions

        return input_sequences, gt_sequences

    def _sampling_from_normal_distribution_pdf(self, mean=0, std=1, amplitude=1):
        distribution = norm(loc=mean, scale=std)
        sampling_values = amplitude * distribution.pdf(self.sampling_time_point)
        return sampling_values

    def _sampling_from_derivative_of_normal_distribution_pdf(self, mean=0, sd=1, amplitude=1):
        _sampling_values = self._sampling_from_normal_distribution_pdf(mean, sd, amplitude)
        sampling_values = (mean - self.sampling_time_point) / (sd * sd) * _sampling_values
        return sampling_values

    def _append_zero_margins(self, data, left_margin_len, right_margin_len):
        left_margin = np.zeros(shape=left_margin_len)
        right_margin = np.zeros(shape=right_margin_len)
        return np.concatenate([left_margin, data, right_margin], axis=0)

    def _add_noise(self, data, noise_ratio):
        max_value = np.max(data, axis=1)
        max_noise = noise_ratio * max_value
        noise = np.random.uniform(low=-1 * max_noise, high=max_noise, size=(self.bs, self.trail_sampling_point_num))
        return data + noise[:, :, np.newaxis]


if __name__ == "__main__":
    # rv = norm(loc=0, scale=1)
    # print(rv.pdf(0), rv.pdf(1))
    # print(np.arange(-0.75, 0.75 + 0.02, step=0.02))
    bs = 16
    data_generator = DataGenerator(bs)
    inputs, gts = data_generator.next_batch()
    print(inputs.shape, gts.shape)
    x = range(175)

    for i in range(bs):
        plt.subplot(4, 4, i+1)
        plt.plot(x, inputs[i, :, 0])
        plt.plot(x, inputs[i, :, 1])
        plt.plot(x, gts[i, :, 0] * 75)
    plt.show()

