import tensorflow as tf


class RNN(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.model = tf.keras.Sequential([])

    def _calc_loss(self, pred, ground_truth):
        pass

    # Return: grads and vars
    def build_grads(self, inputs, ground_truth):
        trainable_vars = self.model.trainable_variables
        with tf.GradientTape() as tape:
            pred = self.model(inputs)
            loss = self._calc_loss(pred, ground_truth)
        grads = tape.gradient(loss, trainable_vars)
        return zip(grads, trainable_vars)

