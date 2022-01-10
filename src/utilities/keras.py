import time
from functools import reduce

import mlflow
import numpy as np
from keras.utils.layer_utils import count_params
from tensorflow import keras


def get_total_parameters(model):
    """
    Get the number of trainable and non trainable parameters
    :param model: The Keras model
    :return: pair trainable_count, non trainable_count
    """
    if hasattr(model, '_collected_trainable_weights'):
        trainable_count = count_params(model._collected_trainable_weights)
    else:
        trainable_count = count_params(model.trainable_weights)

    non_trainable_count = count_params(model.non_trainable_weights)
    return trainable_count, non_trainable_count


class LogCallback(keras.callbacks.Callback):
    def __init__(self, log, frequency, batch_slice=(500, 520)):
        """

        :param log: log object
        :param writer: Summary Writer
        :param frequency: frequency of logging batches (too much frequency will slow down the process)
        """
        super().__init__()
        self.trace = False
        self.trace_finished = False
        self._batch_start_time = None
        self.batch_slice = batch_slice
        self.log = log
        self.batch_times = []
        self.log_frequency = frequency
        self.train_start = None

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        self.train_start = time.perf_counter()
        self.log.info("Starting training - got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        mlflow.log_metric('batch_time', self.get_batch_time())
        mlflow.log_metric('training_time', time.perf_counter() - self.train_start)
        self.log.info("End training")

    def on_epoch_begin(self, epoch, logs=None):
        self.log.info('Epoch: {} '.format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        msg = reduce(lambda a, b: a + b, ["{}: {},\t".format(key, value) for key, value in logs.items()])
        mlflow.log_metrics(logs, step=epoch)
        self.log.info("End epoch {} of training - {}".format(epoch, msg))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        self.log.info("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        self.log.info("Stop testing; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        if not self.trace_finished:
            if self.trace or \
                    (self.batch_slice[0] <= batch <= self.batch_slice[1]):
                self._batch_start_time = time.perf_counter()
                self.trace = True
            elif batch > self.batch_slice[1]:
                self.trace_finished = False
                self.trace = False

    def on_train_batch_end(self, batch, logs=None):
        if self.trace:
            batch_run_time = time.perf_counter() - self._batch_start_time
            self.batch_times.append(batch_run_time)

    def get_batch_time(self):
        return np.mean(self.batch_times)

    def on_test_batch_end(self, batch, logs=None):
        if batch % self.log_frequency == 0:
            msg = reduce(lambda a, b: a + b, ["{}: {},\t".format(key, value) for key, value in logs.items()])
            self.log.info("Batch {} \t - {}".format(batch, msg))