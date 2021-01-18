training_batch_size = 128
learning_rate = 0.01
learning_rate_decay = 0.98
num_epochs = 20
steps_per_epoch = 1000
rnn_units = 128 # 512
save_epochs = 10
cell_type = "LSTM" # RNN / GRU / LSTM
opt_type = "SGD" # Adam / SGD


def get_dir(v_noise_sigma=0, a_noise_sigma=0, delay=0):
    name = "bs" + str(training_batch_size)
    name += "v_noise" + str(v_noise_sigma)
    name += "a_noise" + str(a_noise_sigma)
    name += "_delay" + str(delay)
    name += "_steps" + str(steps_per_epoch)
    name += "_lr" + str(learning_rate)
    name += "_lrDecay" + str(learning_rate_decay)
    name += "_" + cell_type + str(rnn_units)
    name += "_" + opt_type
    name += "/"
    return name
