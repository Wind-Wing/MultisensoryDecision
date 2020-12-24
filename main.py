from network import RNN
import analyse
import constants


def train():
    # TODO: Let the network quickly oscillation
    # TODO: How to let the network emerge the behavior of the brain?
    model = RNN()
    model.train(noise_ratio=0.1, velocity_input_delay=0)


def analyse():
    # Load model
    model = RNN()
    model.load(constants.num_epochs, noise_ratio=0.1, delay=0, lr=constants.learning_rate, decay=constants.learning_rate_decay)

    # Visualize
    analyse.visualize(model, 4)

    # Dynamic system analyse
    analyse.dynamic_system(model, 4)


if __name__ == "__main__":
    train()
    # analyse()
