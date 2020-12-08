import tensorflow as tf
import numpy as np
from network import RNN
import constants


def train():
    model = RNN()

    writer = tf.summary.create_file_writer("./logs/")
    for epoch in range(constants.num_epochs):
        # TODO: Let the network quickly oscillation
        # TODO: How to let the network emerge the behavior of the brain?

        batch_loss = model.train_one_iteration()
        mean_loss = float(np.mean(batch_loss))

        with writer.as_default():
            tf.summary.scalar("loss", mean_loss, step=epoch)
        writer.flush()

        if epoch % constants.save_epochs == 0:
            model.save(epoch)

        if epoch % constants.report_epochs == 0:
            print("%d : loss = %f" % (epoch, mean_loss))


if __name__ == "__main__":
    train()
