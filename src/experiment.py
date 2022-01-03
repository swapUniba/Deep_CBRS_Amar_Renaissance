from ruamel.yaml import YAML
from easydict import EasyDict
from os.path import join as path_join
from time import strftime

from utilities import data
from utilities.utils import LogCallback, get_total_parameters
from models.gnn import BasicGNN

import tensorflow as tf
import numpy as np
import pandas as pd

import logging
import models
import inspect
import os

from utilities.metrics import top_k_metrics

PARAMS_PATH = 'config.yaml'
LOG_FREQUENCY = 100


class Experimenter:
    def __init__(self, config: str):
        """
        Encapsulates all objects needed and performs the train and the evaluation

        :param config: Path of config file
        """
        with open(config, 'r') as params_file:
            yaml = YAML()
            config_str = params_file.read()
            self.config = EasyDict(**yaml.load(config_str))

        self.exp_name = \
            strftime("%m_%d-%H_%M") + '-' + \
            self.config.model.name + '-' + \
            self.config.dataset.load_function_name + '-' + \
            self.config.details

        self.config.dest = path_join(self.config.dest, self.exp_name)
        os.makedirs(self.config.dest, exist_ok=True)

        # Logging stuff
        file_handler = logging.FileHandler(path_join(self.config.dest, 'log.txt'))
        logging.basicConfig(
            handlers=[
                file_handler,
            ],
            format="%(asctime)s %(message)s",
            datefmt='[%H:%M:%S]',
            level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.callback_logger = logging.getLogger('callback')
        self.logger.log(logging.INFO, 'CONFIG \n' + config_str + '\n')

        # Tensorboard
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.config.dest, histogram_freq=LOG_FREQUENCY)
        self.board_writer = tf.summary.create_file_writer(self.config.dest + "/metrics")
        self.board_writer.set_as_default()

        self._retrieve_classes()
        self.trainset = None
        self.testset = None
        self.model = None
        self.parameters = self.config.parameters

    def _retrieve_classes(self):
        """
        Retrieve object classes from strings
        """
        self.config.optimizer_class = getattr(tf.keras.optimizers, self.config.parameters.optimizer.name)

        model_module, model_class = self.config.model.name.split('.')
        model_package = __import__(models.__name__, fromlist=[model_module])
        model_module = getattr(model_package, model_module)
        self.config.model_class = getattr(model_module, model_class)

        self.config.load_function = getattr(data, self.config.dataset.load_function_name)

    def build_dataset(self):
        """
        Builds dataset from configs
        """
        # Get parameters from load method
        init_parameters = inspect.signature(self.config.load_function).parameters
        parameters = {k: self.config.dataset[k]
                      for k in self.config.dataset.keys() & init_parameters.keys()}
        self.trainset, self.testset = self.config.load_function(**parameters)

    def build_model(self):
        """
        Builds model from config parameters
        """
        self.logger.info('Building model...')

        # Additional parameter for GNNs
        if issubclass(self.config.model_class, BasicGNN):
            self.model = self.config.model_class(self.trainset.adj_matrix, **self.config.model)
        else:
            self.model = self.config.model_class(**self.config.model)

        optimizer = self.config.optimizer_class(
            learning_rate=self.parameters.optimizer.lr,
            beta_1=self.parameters.optimizer.beta
        )
        self.model.compile(
            loss=self.parameters.loss,
            optimizer=optimizer,
            metrics=self.parameters.metrics
        )
        # One prediction is needed to build the model
        self.model.predict(self.trainset[0][0])
        self.model.summary(print_fn=self.logger.info, expand_nested=True)
        with self.board_writer.as_default():
            trainable, non_trainable = get_total_parameters(self.model)
            tf.summary.scalar('trainable_params', trainable, step=0)
            tf.summary.scalar('non_trainable_params', non_trainable, step=0)


    def train(self):
        """
        Trains a model on the given parameters, and saves it
        """

        self.logger.info("Experiment folder: " + self.config.dest)

        self.build_dataset()
        self.build_model()

        self.logger.info('Training:')
        history = self.model.fit(
            self.trainset,
            epochs=self.parameters.epochs,
            workers=self.config.n_workers,
            callbacks=[self.tensorboard, LogCallback(self.callback_logger, LOG_FREQUENCY)])

        # creates a HDF5 file 'model.h5'
        self.logger.info('Saving model...')
        save_path = path_join(self.config.dest, 'model.h5')
        self.model.save_weights(save_path)
        self.logger.info('Succesfully saved in ' + save_path)

    def evaluate(self):
        """
        Evaluates the trained model
        """
        self.model.evaluate(self.testset, callbacks=[self.tensorboard])

        # Compute Precision, Recall and F1 @K metrics
        predictions = self.model.predict(self.testset)
        ratings_pred = np.concatenate([self.testset.ratings[:, [0, 1]], predictions], axis=1)
        precision_at, recall_at, f1_at = {}, {}, {}
        ks = [5, 10, 20]
        for k in ks:
            (precision_at[k], recall_at[k], f1_at[k]) = top_k_metrics(self.testset.ratings, ratings_pred, k=k)

        metrics = pd.DataFrame([precision_at, recall_at, f1_at], index=['precision_at', 'recall_at', 'f1_at'])
        self.logger.info('\n' + str(metrics))
        with self.board_writer.as_default():
            for k in ks:
                tf.summary.scalar('precision_at', precision_at[k], step=k)
                tf.summary.scalar('recall_at', recall_at[k], step=k)
                tf.summary.scalar('f1_at', f1_at[k], step=k)

    def run(self):
        """
        Run a full experiment (train and evaluation)
        :return:
        """
        self.train()
        self.evaluate()


if __name__ == "__main__":

    experimenter = Experimenter(PARAMS_PATH)
    experimenter.run()
