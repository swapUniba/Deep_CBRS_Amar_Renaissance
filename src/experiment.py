import keras
from ruamel.yaml import YAML
from easydict import EasyDict
from os.path import join as path_join
from time import strftime

from utilities import data
from models.gnn import BasicGNN

import tensorflow as tf
import numpy as np

import logging
import models
import inspect
import os

from utilities.metrics import top_k_metrics

PARAMS_PATH = 'config.yaml'


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
        logging.basicConfig(
            handlers=[
                logging.FileHandler(path_join(self.config.dest, 'log.txt')),
                logging.StreamHandler()
            ],
            format="%(message)s",
            level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.config.dest, histogram_freq=1)
        self.logger.log(logging.INFO, 'CONFIG \n' + config_str + '\n')

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
        init_parameters = inspect.signature(self.config.model_class.__init__).parameters
        parameters = {k: self.config.model[k]
                      for k in self.config.model.keys() & init_parameters.keys()}

        # Additional parameter for GNNs
        if issubclass(self.config.model_class, BasicGNN):
            self.model = self.config.model_class(self.trainset.adj_matrix, **parameters)
        else:
            self.model = self.config.model_class(**parameters)

    def train(self):
        """
        Trains a model on the given parameters, and saves it
        """

        self.logger.info("Experiment folder: " + self.config.dest)

        self.build_dataset()
        self.build_model()

        self.logger.info('Training:')
        optimizer = self.config.optimizer_class(
            learning_rate=self.parameters.optimizer.lr,
            beta_1=self.parameters.optimizer.beta
        )
        self.model.compile(
            loss=self.parameters.loss,
            optimizer=optimizer,
            metrics=self.parameters.metrics
        )
        history = self.model.fit(
            self.trainset,
            epochs=self.parameters.epochs,
            workers=self.config.n_workers,
            callbacks=[self.tensorboard])

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
        self.logger.info('P@ 5, R@ 5, F@ 5: {}'.format(top_k_metrics(self.testset.ratings, ratings_pred, k=5)))
        self.logger.info('P@10, R@10, F@10: {}'.format(top_k_metrics(self.testset.ratings, ratings_pred, k=10)))
        self.logger.info('P@20, R@20, F@20: {}'.format(top_k_metrics(self.testset.ratings, ratings_pred, k=20)))

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
