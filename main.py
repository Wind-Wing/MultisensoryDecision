from network import RNN
import analyse
import constants


def train():
    # TODO: Let the network quickly oscillation
    # TODO: How to let the network emerge the behavior of the brain?
    model = RNN()
    model.train(v_noise_sigma=0, a_noise_sigma=0, velocity_input_delay=0)


if __name__ == "__main__":
    train()
