import numpy as np
import matplotlib
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
        self.velocity_mean = self.stimulus_duration / 2.

        self.max_noise_amplitude = self.max_velocity_amplitude
        # TODO: validate sigma diff between v and a max. And compare the max value ratio

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
        # TODO: save the training samples. Or just create a static data set.
        # TODO: noise signal ratio.
        # TODO: do I need give time for the RNN to process

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

        # Inputs
        v_sequences = np.reshape(v_sequences, newshape=(self.bs, self.sampling_point_num, 1))
        a_sequences = np.reshape(a_sequences, newshape=(self.bs, self.sampling_point_num, 1))
        input_sequences = np.concatenate([v_sequences, a_sequences], axis=-1)

        # Ground Truths
        # TODO: Does gt need to be the results of discretization add up or integral of continued function?
        # TODO: Avoid the value of gt goes too high -> normalize. Combine with active function
        # TODO: How does brain represents such activity?
        gt_sequences = np.cumsum(v_sequences, axis=1) + np.cumsum(np.abs(a_sequences), axis=1)

        # Margin
        left_margins_len = np.random.randint(low=0, high=self.empty_point_num+1, size=self.bs)
        right_margins_len = self.empty_point_num - left_margins_len
        input_sequences = list(map(self._append_empty_margin, input_sequences, left_margins_len, right_margins_len))
        gt_sequences = list(map(self._append_empty_margin, gt_sequences, left_margins_len, right_margins_len))

        # Directions
        directions = np.random.uniform(low=0, high=1, size=(self.bs, 1, 1)) > 0.5
        input_sequences = input_sequences * directions
        gt_sequences = gt_sequences * directions

        return input_sequences, gt_sequences

    def _sampling_from_normal_distribution_pdf(self, mean=0, std=1, amplitude=1):
        distribution = norm(loc=mean, scale=std)
        sampling_values = amplitude * distribution.pdf(self.sampling_time_point)
        return sampling_values

    def _sampling_from_derivative_of_normal_distribution_pdf(self, mean=0, sd=1, amplitude=1):
        _sampling_values = self._sampling_from_normal_distribution_pdf(mean, sd, amplitude)
        sampling_values = (mean - self.sampling_time_point) / (sd * sd) + _sampling_values
        return sampling_values

    def _append_empty_margin(self, data, left_margin_len, right_margin_len):
        channel = int(data.shape[-1])
        left_margin = np.zeros(shape=[left_margin_len, channel])
        right_margin = np.zeros([right_margin_len, channel])
        return np.concatenate([left_margin, data, right_margin], axis=0)


if __name__ == "__main__":
    # rv = norm(loc=0, scale=1)
    # print(rv.pdf(0), rv.pdf(1))
    # print(np.arange(-0.75, 0.75 + 0.02, step=0.02))

    data_generator = DataGenerator(4)
    inputs, gts = data_generator.next_batch()
    print(inputs)
    print(gts)
