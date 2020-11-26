class DataGenerator(object):
    def __init__(self, batch_size):
        # TODO: adjust the params
        self.bs = batch_size
        self.time_interval = 0.2
        self.time_length = 4
        self.sample_point = self.time_length / self.time_interval

    def next_batch(self, training_flag):
        # TODO: define data model
        # TODO: change model according to exp setting
        # TODO: save the training sample. Or just create a static data set
        pass
