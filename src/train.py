from tensorflow import keras
import models
from ruamel.yaml import YAML
from easydict import EasyDict
from os.path import join as path_join

import logging
import keras
import datasets
import inspect

PARAMS_PATH = 'config.yaml'


class Trainer:
    def __init__(self, config: EasyDict, logger):
        """
        Encapsulates all objects needed and performs the train

        :param config: EasyDict of all parameters
        :param logger: loggger object
        """
        self.config = config
        self._retrieve_classes()
        self.logger = logger
        self.dataset = None

    def _retrieve_classes(self):
        """
        Retrieve object classes from strings
        """
        self.config.optimizer = getattr(keras.optimizers, config.parameters.optimizer)
        self.config.model_class = getattr(models, self.config.model_name)
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

    def train(self):
        """
        Trains a model on the given parameters, and saves it
        """
        print(self.config.user_source)
        print(self.config.item_source)
        print(self.config.dest)
        print(self.config.prediction_dest)

        print('Training:')
        model = self.config.model_class(feature_based=False)
        optimizer = self.config.optimizer(
            learning_rate=config.parameters.optimizer.lr,
            beta_1=config.parameters.optimizer.beta
        )
        model.compile(
            loss=config.parameters.loss,
            optimizer=optimizer,
            metrics=config.parameters.metrics
        )
        model = model.fit(self.dataset, epochs=config.parameters.lr, workers=config.n_workers)

        # creates a HDF5 file 'model.h5'
        self.logger.info('Saving model...')
        save_path = path_join(config.dest, 'model.h5')
        model.save(save_path)
        self.logger.info('Succesfully saved in ' + save_path)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    with open(PARAMS_PATH, 'r') as params_file:
        yaml = YAML()
        config = EasyDict(**yaml.load(params_file))

    trainer = Trainer(config, logger)
    trainer.build_dataset()
    trainer.train()