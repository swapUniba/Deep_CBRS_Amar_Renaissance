from ruamel.yaml import YAML
from easydict import EasyDict
from os.path import join as path_join
from time import strftime

from utilities import datasets

import tensorflow as tf
import logging
import models
import inspect
import os

PARAMS_PATH = 'config.yaml'


class Trainer:
    def __init__(self, config: str):
        """
        Encapsulates all objects needed and performs the train

        :param config: Path of config file
        """
        with open(PARAMS_PATH, 'r') as params_file:
            yaml = YAML()
            config_str = params_file.read()
            self.config = EasyDict(**yaml.load(config_str))

        self.exp_name = \
            strftime("%m_%d-%H_%M") + '-' + \
            self.config.model.name + '-' + \
            self.config.dataset.name + '-' + \
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
        self.logger.log(logging.INFO, 'CONFIG \n' + config_str + '\n')

        self._retrieve_classes()
        self.dataset = None
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

        self.config.dataset_class = getattr(datasets, self.config.dataset.name)

    def build_dataset(self):
        """
        Builds dataset from configs
        """
        self.logger.info('Building dataset...')
        init_parameters = inspect.signature(self.config.dataset_class.__init__).parameters
        parameters = {k: self.config.dataset[k]
                      for k in self.config.dataset.keys() & init_parameters.keys()}
        self.dataset = self.config.dataset_class(**parameters)

    def build_model(self):
        """
        Builds model from config parameters
        """
        self.logger.info('Building model...')
        init_parameters = inspect.signature(self.config.model_class.__init__).parameters
        parameters = {k: self.config.model[k]
                      for k in self.config.model.keys() & init_parameters.keys()}
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
            self.dataset,
            epochs=self.parameters.epochs,
            workers=self.config.n_workers)

        # creates a HDF5 file 'model.h5'
        self.logger.info('Saving model...')
        save_path = path_join(self.config.dest, 'model.h5')
        tf.saved_model.save(self.model, save_path)
        self.logger.info('Succesfully saved in ' + save_path)


if __name__ == "__main__":

    trainer = Trainer(PARAMS_PATH)
    trainer.train()
