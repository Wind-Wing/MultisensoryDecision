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

        self.x = tf.keras.layers.Input(shape=self.data_generator.get_inputs_shape(), batch_size=constants.training_batch_size)
        self.state_sequences = tf.keras.layers.SimpleRNN(
            units=constants.rnn_units,
            return_sequences=True)(self.x)
        y = tf.keras.layers.Dense(1)(self.state_sequences)

        self.model = tf.keras.Model(inputs=self.x, outputs=y)
        self.model.summary()

        self.model.compile(
            optimizer=tf.optimizers.Adam(lr=constants.learning_rate, decay=constants.learning_rate_decay),
            loss=tf.keras.losses.MeanSquaredError()
        )

    def train(self, noise_ratio=None, velocity_input_delay=0):
        sub_dir = self.get_dir(noise_ratio=noise_ratio, delay=velocity_input_delay)
        print(sub_dir)
        os.mkdir(self.ckpt_dir + sub_dir)

        tensor_board = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir + sub_dir, update_freq=100)
        model_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=self.ckpt_dir + sub_dir + self.ckpt_name)
        pred_visulize = tf.keras.callbacks.LambdaCallback(on_epoch_end= lambda _, __: analyse.visualize(self.model, 4))

        self.model.fit(
            x=self.data_generator.batch_generator(noise_ratio, velocity_input_delay),
            epochs=constants.num_epochs,
            steps_per_epoch=constants.steps_per_epoch,
            verbose=1,
            workers=4,
            use_multiprocessing=True,
            callbacks=[tensor_board, model_ckpt, pred_visulize]
        )

    def load(self, epoch, noise_ratio=None, delay=0, lr=constants.learning_rate, decay=constants.learning_rate_decay):
        ckpt_path = self.ckpt_dir + self.get_dir(
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

    def get_dir(self, noise_ratio=None, delay=0, lr=constants.learning_rate, decay=constants.learning_rate_decay):
        name = "bs" + str(constants.training_batch_size)
        if noise_ratio is not None:
            name += "_noise" + str(noise_ratio)
        name += "_delay" + str(delay)
        name += "_steps" + str(constants.steps_per_epoch)
        name += "_lr" + str(lr)
        name += "_lrDecay" + str(decay)
        name += "/"
        return name

    def get_rnn_states(self, inputs, bs):
        rnn_states_model = tf.keras.Model(self.x, self.state_sequences)
        rnn_states = rnn_states_model.predict(
            x=inputs,
            batch_size=bs
        )
        return rnn_states
