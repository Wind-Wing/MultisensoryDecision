import tensorflow as tf
import constants
from network import RNN


# TODO: Go through the code
def train():
    model = RNN()

    writer = tf.summary.create_file_writer("./logs/")
    for epoch in range(constants.num_epochs):
        loss = model.train_one_iteration()
        with writer.as_default():
            tf.summary.scalar("training_loss", loss, step=epoch)
        writer.flush()


if __name__ == "__main__":
    train()
