import tensorflow as tf
import constants
from data import DataGenerator
from network import RNN

def train():
    training_flag = True
    batch_size = tf.Variable(constants.training_batch_size, dtype=tf.int32, trainable=False, name='batch_size')
    optimizer = tf.optimizers.Adam(lr=constants.training_batch_size, decay=constants.learning_rate_decay)
    model = RNN()

    for epoch in range(num_epochs):
        data_generator = DataGenerator(batch_size, training_flag)
        grads_and_vars = model.build_grads(inputs, ground_truth)
        optimizer.apply_gradients(grads_and_vars)

        # Tensorboard




