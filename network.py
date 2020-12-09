import time
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
        # TODO: Momenta update states
        # TODO: Dropout
        # TODO: Active function
        # TODO: Regularization

        self.ckpt_prefix = "./ckpts/checkpoint-"
        self.log_path = "./logs/"
        self.data_generator = DataGenerator(constants.training_batch_size)

        x = tf.keras.layers.Input(shape=self.data_generator.get_inputs_shape(), batch_size=constants.training_batch_size)
        self.state_sequences = tf.keras.layers.SimpleRNN(
            units=constants.rnn_units,
            return_sequences=True)(x)
        y = tf.keras.layers.Dense(1)(self.state_sequences)

        self.model = tf.keras.Model(input=x, output=y)
        self.model.summary()

        self.model.compile(
            optimizer=tf.optimizers.Adam(lr=constants.training_batch_size, decay=constants.learning_rate_decay),
            loss=tf.keras.losses.mse()
        )

    def train(self):
        self.model.fit(
            x=self.data_generator.batch_generator(),
            epochs=constants.num_epochs,
            verbose=2,
            workers=4,
            use_multiprocessing=True,
            callbacks=[tf.keras.callbacks.TensorBoard(log_dir=self.log_path)]
        )

    def save(self, epoch):
        self.model.save(self.ckpt_prefix + str(epoch))

    def load(self):
        self.model.load(tf.train.latest_checkpoint(self.ckpt_prefix))

    def evaluate(self):
        pass

