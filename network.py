import tensorflow as tf
import constants
from data import DataGenerator


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

        self.optimizer = tf.optimizers.Adam(lr=constants.training_batch_size, decay=constants.learning_rate_decay)
        self.saver = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_dir = "./ckpts"

        batch_size = tf.Variable(constants.training_batch_size, dtype=tf.int32, trainable=False, name='batch_size')
        self.data_generator = DataGenerator(batch_size)

    def calc_loss(self, pred_list, gt_list):
        # TODO: Regularizer
        return tf.losses.mse(pred_list, gt_list)

    # Return: grads and vars
    def _build_grads(self, input_list, gt_list):
        trainable_vars = self.model.trainable_variables
        with tf.GradientTape() as tape:
            pred = self.model(input_list)
            loss = self.calc_loss(pred, gt_list)
        grads = tape.gradient(loss, trainable_vars)
        return zip(grads, trainable_vars)

    def train_one_iteration(self):
        input_list, gt_list = self.data_generator.next_batch(training_flag=True)
        grads_and_vars = self._build_grads(input_list, gt_list)
        self.optimizer.apply_gradients(grads_and_vars)

    def save(self):
        self.saver.save(self.ckpt_dir)

    def load(self):
        self.saver.load(tf.train.latest_checkpoint(self.ckpt_dir))

    def evaluate(self):
        pass

