from ruamel.yaml import YAML
from easydict import EasyDict
from os.path import join as path_join
from time import strftime

from utilities.utils import \
    get_experiment_logger, nested_dict_update, make_grid, mlflow_linearize, setup_mlflow
from utilities.keras import get_total_parameters, LogCallback
from models.basic import BasicRS, BasicGNN
from models.hybrid import HybridCBRS, HybridBertGNN
from models.kgnn import KGATCallback
from utilities.metrics import top_k_predictions, top_k_metrics
from utilities import losses
from data import loaders

import tensorflow as tf
import numpy as np
import pandas as pd

import logging
import models
import inspect
import os
import io
import mlflow


PARAMS_PATH = 'config.yaml'
EXPERIMENTS_PATH = 'experiments.yaml'
MLFLOW_PATH = './mlruns'
MLFLOW_EXP_NAME = 'SIS - Movielens-1M - BasicRS with GNNs'
LOG_FREQUENCY = 100
METRICS_TOP_KS = [5, 10]
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
EXP_PATH = setup_mlflow(MLFLOW_EXP_NAME, MLFLOW_PATH)


class Experimenter:
    def __init__(self, config):
        """
        Encapsulates all objects needed and performs the train and the evaluation

        :param config: dict of configuration parameters
        """
        self.config = EasyDict(**config)
        # Set the random seed
        tf.random.set_seed(self.config.seed)

        # Initialize the experiment name
        self.exp_name = \
            strftime("%m_%d-%H_%M") + '-' + \
            self.config.model.name
        if BasicRS.__name__ in self.config.model.name:
            pass
        elif HybridCBRS.__name__ in self.config.model.name:
            if self.config.model.feature_based:
                self.exp_name = self.exp_name + '-' + 'feature'
            else:
                self.exp_name = self.exp_name + '-' + 'entity'
        else:  # GNN-based models
            self.exp_name = self.exp_name + '-' + \
                str(self.config.model.l2_regularizer) + '-' + \
                self.config.model.final_node

        if self.config.details is not None and self.config.details != '':
            self.exp_name = self.exp_name + '-' + self.config.details
        mlflow.start_run(run_name=self.exp_name)
        mlflow.log_params(mlflow_linearize(config))

        self.config.dest = path_join(EXP_PATH, mlflow.active_run().info.run_id, 'artifacts')
        self.predictions_dest = path_join(self.config.dest, "predictions")
        os.makedirs(self.config.dest, exist_ok=True)
        os.makedirs(self.predictions_dest, exist_ok=True)

        # Save the config in the experiment folder (R E P R O D U C I B I L I T Y)
        with open(path_join(self.config.dest, "config.yaml"), 'w') as config_output:
            YAML().dump(config, config_output)

        # Logging init
        self.logger = get_experiment_logger(self.config.dest)

        # Print config
        self.logger.log(logging.INFO, 'CONFIG \n')
        config_str = io.StringIO()
        YAML().dump(config, config_str)
        config_str = config_str.getvalue()
        self.logger.info('CONFIG')
        self.logger.info(config_str)

        self._retrieve_classes()
        self.trainset = None
        self.testset = None
        self.model = None
        self.optimizer = None
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

        self.config.load_function = getattr(loaders, self.config.dataset.load_function_name)

    def build_dataset(self):
        """
        Builds dataset from configs
        """
        # Get parameters from load method
        init_parameters = inspect.signature(self.config.load_function).parameters
        parameters = {k: self.config.dataset[k]
                      for k in self.config.dataset.keys() & init_parameters.keys()}
        self.trainset, self.testset = self.config.load_function(**parameters)

    def build_optimizer(self):
        init_parameters = inspect.signature(self.config.optimizer_class).parameters
        parameters = {k: self.config.parameters.optimizer[k]
                      for k in self.config.parameters.optimizer.keys() & init_parameters.keys()}
        self.optimizer = self.config.optimizer_class(**parameters)

    def build_model(self):
        """
        Builds model from config parameters
        """
        self.logger.info('Building model...')

        # Additional parameter for GNNs
        if issubclass(self.config.model_class, BasicGNN) or issubclass(self.config.model_class, HybridBertGNN):
            self.model = self.config.model_class(self.trainset.adj_matrix, **self.config.model)
        else:
            self.model = self.config.model_class(**self.config.model)

        # Instantiate custom loss
        if hasattr(losses, self.parameters.loss):
            self.parameters.loss = getattr(losses, self.parameters.loss)()
        # Compile the model
        self.model.compile(
            loss=self.parameters.loss,
            optimizer=self.optimizer,
            metrics=self.parameters.metrics
        )

        # One prediction is needed to build the model
        self.model(self.trainset[0][0])
        self.model.summary(print_fn=self.logger.info, expand_nested=True)
        trainable, non_trainable = get_total_parameters(self.model)
        mlflow.log_metrics({'trainable_params': trainable, 'non_trainable_params': non_trainable})

    def train(self):
        """
        Trains a model on the given parameters, and saves it
        """

        self.logger.info("Experiment folder: " + self.config.dest)

        self.build_dataset()
        self.build_optimizer()
        self.build_model()

        # Set the callbacks
        callbacks = [LogCallback(self.logger, LOG_FREQUENCY)]
        if 'KGAT' in self.model.__class__.__name__:
            callbacks.append(KGATCallback(self.logger))

        self.logger.info('Training:')
        self.model.fit(
            self.trainset,
            epochs=self.parameters.epochs,
            workers=self.config.n_workers,
            callbacks=callbacks
        )

        # creates a HDF5 file 'model.h5'
        self.logger.info('Saving model...')
        save_path = path_join(self.config.dest, 'model.h5')
        self.model.save_weights(save_path)
        self.logger.info('Succesfully saved in ' + save_path)

    def evaluate(self):
        """
        Evaluates the trained model
        """
        self.model.evaluate(self.testset)

        # Compute Precision, Recall and F1 @K metrics
        predictions = self.model.predict(self.testset)
        ratings_pred = np.concatenate([self.testset.ratings[:, [0, 1]], predictions], axis=1)
        precision_at, recall_at, f1_at = {}, {}, {}

        # Compute and log metrics @K
        for k in METRICS_TOP_KS:
            # Compute the top predictions and save them to file
            top_predictions = top_k_predictions(ratings_pred, self.trainset.users, self.trainset.items, k=k)
            top_k_dest = path_join(self.predictions_dest, "top_{}".format(k))
            os.makedirs(top_k_dest, exist_ok=True)
            top_predictions.to_csv(path_join(top_k_dest, "predictions_1.tsv"), sep='\t', header=False, index=False)

            # Compute the metrics given the filepaths of test set and the top predictions path
            top_k_metrics(self.config.dataset.test_ratings_filepath, top_k_dest)
            results = pd.read_csv(path_join(top_k_dest, "results.tsv"), sep='\t', header=None)
            results = results.drop(0, axis=1).to_numpy().squeeze()
            precision_at[k], recall_at[k], f1_at[k] = results[0], results[1], results[2]

            # Log metrics using MLFlow
            mlflow.log_metrics({"precision_at_{}".format(k): precision_at[k],
                                "recall_at_{}".format(k): recall_at[k],
                                "f1_at_{}".format(k): f1_at[k]
                                })

        # Save metrics also in raw log
        metrics = pd.DataFrame([precision_at, recall_at, f1_at], index=['precision_at', 'recall_at', 'f1_at'])
        self.logger.info('\n' + str(metrics))

    def run(self):
        """
        Run a full experiment (train and evaluation)
        """
        self.train()
        self.evaluate()
        self.close()

    def close(self):
        """
        Close all streams
        """
        for handler in self.logger.handlers:
            handler.close()
        mlflow.end_run()


class MultiExperimenter:
    """
    Runs multiple experiments by reading an experiment file and overriding parameters from a base config
    """

    def __init__(self):
        # Loads the base config
        with open(PARAMS_PATH, 'r') as params_file:
            yaml = YAML()
            config_str = params_file.read()
            self.base_config = yaml.load(config_str)

        # Loads the experiments file
        with open(EXPERIMENTS_PATH, 'r') as params_file:
            yaml = YAML()
            config_str = params_file.read()
            config = yaml.load(config_str)

        # Get list of experiments
        self.experiments = config.get('linear') if config.get('linear') else {}
        # Get grid of experiments
        dict_lists = config.get('grid')
        if dict_lists:
            for grid in dict_lists.values():
                dicts = make_grid(grid)
                self.experiments = {**self.experiments, **{str(elem): elem for elem in dicts}}
        print("Retrieved experiments:")
        for exp in self.experiments.keys():
            print(exp)

    def run_experiment(self, exp_name):
        """
        Runs a specific experiment
        :param exp_name: Desired experiment to run
        """

        additional_params = self.experiments[exp_name]
        config = self.base_config.copy()
        if additional_params:  # Dict could also be None to run the base config
            config = nested_dict_update(config, additional_params)

        print('-----------------------------------------------\n'
              '{}\n'.format(exp_name),
              '-----------------------------------------------\n'
              )
        exp = Experimenter(config)
        exp.run()

    def run(self):
        """
        Runs all the experiments
        """
        for exp_name in self.experiments:
            self.run_experiment(exp_name)


if __name__ == "__main__":
    exps = MultiExperimenter()
    exps.run()
