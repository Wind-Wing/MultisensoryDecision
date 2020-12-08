import tensorflow as tf
import constants
from network import RNN


# TODO: Go through the code
# TODO: check every weight/state value in one trail
def train():
    model = RNN()

    writer = tf.summary.create_file_writer("./logs/")
    for epoch in range(constants.num_epochs):
        # TODO: Let the network quickly oscillation
        # TODO: How to let the network emerge the behavior of the brain?
        loss = model.train_one_iteration()
        with writer.as_default():
            tf.summary.scalar("loss", loss, step=epoch)
        writer.flush()


if __name__ == "__main__":
    train()
