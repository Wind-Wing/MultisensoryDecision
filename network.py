import time
import os
import tensorflow as tf
import constants
from data import DataGenerator
import analyse


class RNN(object):
    def __init__(self):
        """ Basic RNN: output = new_state = act(W_hi * input + W_hh * state + B) """
        # TODO: Try GRU, LSTM
        # TODO: Add noise
        # TODO: Momenta update states
        # TODO: Dropout
        # TODO: Active function
        # TODO: Regularization

        self.ckpt_dir = "./ckpts/"
        self.ckpt_name = "ckeckpoint-{epoch:d}.hdf5"
        self.log_dir = "./logs/"
        self.data_generator = DataGenerator(constants.training_batch_size)

        if constants.cell_type == "RNN":
            self.cell_type = tf.keras.layers.SimpleRNN
        elif constants.cell_type == "LSTM":
            self.cell_type = tf.keras.layers.LSTM
        elif constants.cell_type == "GRU":
            self.cell_type = tf.keras.layers.GRU
        else:
            raise Exception

        self.x = tf.keras.layers.Input(shape=self.data_generator.get_inputs_shape(), batch_size=constants.training_batch_size)
        self.state_sequences = self.cell_type(
            units=constants.rnn_units,
            return_sequences=True)(self.x)
        y = tf.keras.layers.Dense(1)(self.state_sequences)

        self.model = tf.keras.Model(inputs=self.x, outputs=y)
        self.model.summary()

        if constants.opt_type == "Adam":
            opt = tf.optimizers.Adam(lr=constants.learning_rate)
        elif constants.opt_type == "SGD":
            exp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=constants.learning_rate,
                decay_steps=100,
                decay_rate=constants.learning_rate_decay
            )
            opt = tf.optimizers.SGD(learning_rate=exp_lr)
        else:
            raise Exception

        self.model.compile(
            optimizer=opt,
            loss=tf.keras.losses.MeanSquaredError()
        )

    def train(self, noise_ratio=None, velocity_input_delay=0):
        sub_dir = constants.get_dir(noise_ratio=noise_ratio, delay=velocity_input_delay)
        print(sub_dir)
        _path = self.ckpt_dir + sub_dir
        if not os.path.exists(_path):
            os.mkdir(_path)

        tensor_board = LRTensorBoard(log_dir=self.log_dir + sub_dir, update_freq=100)
        model_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=self.ckpt_dir + sub_dir + self.ckpt_name)
        pred_visualize = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda _, __: analyse.visualize(self.model))

        print("Start Training")
        self.model.fit(
            x=self.data_generator.batch_generator(noise_ratio, velocity_input_delay),
            batch_size=constants.training_batch_size,
            epochs=constants.num_epochs,
            steps_per_epoch=constants.steps_per_epoch,
            verbose=1,
            workers=6,
            use_multiprocessing=True,
            callbacks=[tensor_board, model_ckpt, pred_visualize]
        )

    def load(self, epoch, noise_ratio=None, delay=0, lr=constants.learning_rate, decay=constants.learning_rate_decay):
        ckpt_path = self.ckpt_dir + constants.get_dir(
            noise_ratio=noise_ratio,
            delay=delay,
            lr=lr,
            decay=decay
        )
        ckpt_name = ckpt_path + self.ckpt_name.format(epoch=epoch)
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


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs["lr"] = self.model.optimizer.learning_rate(batch)
        super().on_batch_end(batch, logs)

