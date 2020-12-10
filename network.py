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

        self.ckpt_path = "./ckpts/checkpoint-{epoch:d}-" + str(constants.steps_per_epoch) + ".hdf5"
        self.log_path = "./logs/"
        self.data_generator = DataGenerator(constants.training_batch_size)

        self.x = tf.keras.layers.Input(shape=self.data_generator.get_inputs_shape(), batch_size=constants.training_batch_size)
        self.state_sequences = tf.keras.layers.SimpleRNN(
            units=constants.rnn_units,
            return_sequences=True)(self.x)
        y = tf.keras.layers.Dense(1)(self.state_sequences)

        self.model = tf.keras.Model(inputs=self.x, outputs=y)
        self.model.summary()

        self.model.compile(
            optimizer=tf.optimizers.Adam(lr=constants.training_batch_size, decay=constants.learning_rate_decay),
            loss=tf.keras.losses.MeanSquaredError()
        )

    def train(self):
        tensor_board = tf.keras.callbacks.TensorBoard(log_dir=self.log_path, update_freq=100)
        model_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=self.ckpt_path)

        self.model.fit(
            x=self.data_generator.batch_generator(),
            epochs=constants.num_epochs,
            steps_per_epoch=constants.steps_per_epoch,
            verbose=1,
            workers=4,
            use_multiprocessing=True,
            callbacks=[tensor_board, model_ckpt]
        )

    def load(self, epoch):
        ckpt_name = self.ckpt_path.format(epoch=epoch)
        print("Load from " + ckpt_name)
        print("_________________________________________________________________")
        self.model.load_weights(ckpt_name)

    def predict(self, inputs, bs):
        preds = self.model.predict(
            x=inputs,
            batch_size=bs
        )
        return preds

    def get_rnn_states(self, inputs, bs):
        rnn_states_model = tf.keras.Model(self.x, self.state_sequences)
        rnn_states = rnn_states_model.predict(
            x=inputs,
            batch_size=bs
        )
        return rnn_states



