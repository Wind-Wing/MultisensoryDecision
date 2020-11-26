import constants
from network import RNN


# TODO: Go through the code
def train():
    model = RNN()

    for epoch in range(constants.num_epochs):
        model.train_one_iteration()

        # TODO: Tensorboard


if __name__ == "__main__":
    pass
