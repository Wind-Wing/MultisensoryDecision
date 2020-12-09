import tensorflow as tf
import constants
from data import DataGenerator


class RNN(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.data_generator = DataGenerator(constants.training_batch_size)

        """ Basic RNN: output = new_state = act(W_hi * input + W_hh * state + B) """
        # TODO: Try GRU, LSTM
        # TODO: Add noise
        # TODO: Momenta update states
        # TODO: Dropout
        # TODO: active function

        x = tf.keras.layers.Input(shape=self.data_generator.get_inputs_shape(), batch_size=constants.training_batch_size)
        self.state_sequences = tf.keras.layers.SimpleRNN(
            units=constants.rnn_units,
            return_sequences=True)(x)
        y = tf.keras.layers.Dense(1)(self.state_sequences)

        print(x.shape, self.state_sequences.shape, y.shape)

        self.model = tf.keras.Model(x, y)
        self.model.summary()

        self.optimizer = tf.optimizers.Adam(lr=constants.training_batch_size, decay=constants.learning_rate_decay)
        self.saver = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_prefix = "./ckpts/checkpoint-"

    def _calc_loss(self, pred_list, gt_list):
        # TODO: Regularizer
        return tf.losses.mse(pred_list, gt_list)
        # return tf.math.reduce_mean(tf.losses.mse(pred_list, gt_list), axis=1, keepdims=True)

    # Return: grads and vars
    def _build_grads(self, input_list, gt_list):
        trainable_vars = self.model.trainable_variables
        with tf.GradientTape() as tape:
            pred = self.model(input_list)
            loss = self._calc_loss(pred, gt_list)
        grads = tape.gradient(loss, trainable_vars)
        return loss, zip(grads, trainable_vars)

    def train_one_iteration(self):
        input_list, gt_list = self.data_generator.next_batch()
        loss, grads_and_vars = self._build_grads(input_list, gt_list)
        self.optimizer.apply_gradients(grads_and_vars)
        return loss

    def save(self, epoch):
        self.saver.save(self.ckpt_prefix + str(epoch))

    def load(self):
        self.saver.load(tf.train.latest_checkpoint(self.ckpt_prefix))

    def evaluate(self):
        pass

    def get_learning_rate(self):
        return self.model.optimizer.lr

