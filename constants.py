# TODO: change into args dict
# TODO: read from config file and train one by one
training_batch_size = 128
learning_rate = 0.01
learning_rate_decay = 0.98
num_epochs = 20
steps_per_epoch = 1000
rnn_units = 128
save_epochs = 10
cell_type = "RNN" # RNN / GRU / LSTM
opt_type = "SGD" # Adam / SGD


def get_dir(noise_ratio=None, delay=0, lr=learning_rate, decay=learning_rate_decay):
    name = "bs" + str(training_batch_size)
    if noise_ratio is not None:
        name += "_noise" + str(noise_ratio)
    name += "_delay" + str(delay)
    name += "_steps" + str(steps_per_epoch)
    name += "_lr" + str(lr)
    name += "_lrDecay" + str(decay)
    name += "_" + cell_type + str(rnn_units)
    name += "_" + opt_type
    name += "/"
    return name
