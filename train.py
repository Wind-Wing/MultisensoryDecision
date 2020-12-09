from network import RNN
import pickle


def train():
    # TODO: Let the network quickly oscillation
    # TODO: How to let the network emerge the behavior of the brain?
    model = RNN()
    model.train()


def test():
    model = RNN()
    model.load()
    model.evaluate()


if __name__ == "__main__":
    train()
