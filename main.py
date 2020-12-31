from network import RNN
import analyse
import constants


def train():
    # TODO: Let the network quickly oscillation
    # TODO: How to let the network emerge the behavior of the brain?
    model = RNN()
    model.train(noise_ratio=0.1, velocity_input_delay=0)


if __name__ == "__main__":
    train()
