import tensorflow as tf
import constants

class RNN(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        """ Basic RNN: output = new_state = act(W_hi * input + W_hh * state + B) """
        # TODO: Try GRU, LSTM
        # TODO: Add noise
        # TODO: collect states
        # TODO: Momenta update states
        # TODO: Dropout
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.SimpleRNN(
            units=constants.rnn_units
        ))
        self.model.summary()

    def _calc_loss(self, pred_list, gt_list):
        # TODO: Regularizer
        return tf.losses.mse(pred_list, gt_list)

    # Return: grads and vars
    def build_grads(self, input_list, gt_list):
        trainable_vars = self.model.trainable_variables
        with tf.GradientTape() as tape:
            pred = self.model(input_list)
            loss = self._calc_loss(pred, input_list)
        grads = tape.gradient(loss, trainable_vars)
        return zip(grads, trainable_vars)

    def save(self):
        pass

    def load(self):
        pass

