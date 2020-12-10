from network import RNN
from data import DataGenerator
import constants


def train():
    # TODO: Let the network quickly oscillation
    # TODO: How to let the network emerge the behavior of the brain?
    model = RNN()
    model.train()


def predict():
    model = RNN()
    model.load(constants.num_epochs)

    bs = 16
    data_generator = DataGenerator(batch_size=bs)
    inputs, gts = data_generator.next_batch(noise_ratio=0., velocity_input_delay=0)
    preds = model.predict(inputs, bs)
    stats = model.get_rnn_states(inputs, bs)
    print(preds.shape, stats.shape)


if __name__ == "__main__":
    # train()
    predict()