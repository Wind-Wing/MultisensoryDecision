from network import RNN
from data import DataGenerator
import constants
import matplotlib.pyplot as plt
import math


def train():
    # TODO: Let the network quickly oscillation
    # TODO: How to let the network emerge the behavior of the brain?
    model = RNN()
    model.train(noise_ratio=0.1, velocity_input_delay=0)


def predict(bs=16):
    model = RNN()
    model.load(constants.num_epochs)

    data_generator = DataGenerator(batch_size=bs)
    inputs, gts = data_generator.next_batch(noise_ratio=0., velocity_input_delay=0)
    preds = model.predict(inputs, bs)
    stats = model.get_rnn_states(inputs, bs)
    return inputs, gts, preds, stats


def visualize():
    bs = 1
    fig_size = int(math.sqrt(bs))
    inputs, gts, preds, stats = predict(bs)
    x = range(int(inputs.shape[1]))

    for i in range(bs):
        plt.subplot(fig_size, fig_size, i+1)
        plt.plot(x, inputs[i, :, 0])
        plt.plot(x, inputs[i, :, 1])
        plt.plot(x, gts[i, :, 0] * 75, linestyle='-')
        plt.plot(x, preds[i, :, 0] * 75)
    plt.show()
    print(preds.flatten())
    print(gts.flatten())


if __name__ == "__main__":
    train()
