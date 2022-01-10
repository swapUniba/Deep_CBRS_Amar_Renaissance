import logging
import os
from itertools import groupby, product
from logging import FileHandler, LogRecord

import mlflow
import pandas as pd
import collections


def top_scores(predictions, n):
    top_n_scores = pd.DataFrame()
    for u in list(set(predictions['users'])):
        p = predictions.loc[predictions['users'] == u]
        top_n_scores = top_n_scores.append(p.head(n))
    return top_n_scores


def nested_dict_update(d, u):
    """
    Dictionary update suitable for nested dictionary
    :param d: original dict
    :param u: dict from where updates are taken
    :return: Updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def linearize(dictionary):
    """
    Linearize a nested dictionary making keys, tuples
    :param dictionary: nested dict
    :return: one level dict
    """
    exps = []
    for key, value in dictionary.items():
        if isinstance(value, collections.abc.Mapping):
            exps.extend(((key, lin_key), lin_value) for lin_key, lin_value in linearize(value))
        if isinstance(value, list):
            exps.append((key, value))
    return exps


def extract(elem: tuple):
    """
    Exctract the element of a single element tuple
    :param elem: tuple
    :return: element of the tuple if singleton or the tuple itself
    """
    if len(elem) == 1:
        return elem[0]
    return elem


def delinearize(lin_dict):
    """
    Convert a dictionary where tuples can be keys in na nested dictionary
    :param lin_dict: dicionary where keys can be tuples
    :return:
    """
    # Take keys that are tuples
    filtered = list(filter(lambda x: isinstance(x[0], tuple), lin_dict.items()))
    # Group it to make one level
    grouped = groupby(filtered, lambda x: x[0][0])
    # Create the new dict and apply recursively
    new_dict = {k: delinearize({extract(elem[0][1:]): elem[1] for elem in v}) for k, v in grouped}
    # Remove old items and put new ones
    for key, value in filtered:
        lin_dict.pop(key)
    delin_dict = {**lin_dict, **new_dict}
    return delin_dict


def make_grid(dict_of_list):
    """
    Produce a list of dict for each combination of values in the input dict given by the list of values
    :param dict_of_list: a dictionary where values can be lists
    :return: a list of dictionaries given by the cartesian product of values in the input dictionary
    """
    # Linearize the dict to make the cartesian product straight forward
    linearized_dict = linearize(dict_of_list)
    # Compute the grid
    keys, values = zip(*linearized_dict)
    grid_dict = list(dict(zip(keys, values_list)) for values_list in product(*values))
    # Delinearize the list of dicts
    return [delinearize(dictionary) for dictionary in grid_dict]


class FlushFileHandler(FileHandler):
    def emit(self, record: LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_mlflow(artifact_path):
    """
    """
    mlflow.tensorflow.autolog()
    os.makedirs(artifact_path, exist_ok=True)
    os.makedirs(os.path.join(artifact_path, '.trash'), exist_ok=True)

    experiment = mlflow.get_experiment_by_name('SIS')
    if not experiment:
        exps = os.listdir(artifact_path)
        exps.pop(exps.index('.trash'))
        if len(exps) == 0:
            exp_id = '0'
        else:
            exp_id = str(max([int(exp) for exp in exps]))
        experiment_id = mlflow.create_experiment(
            'SIS', artifact_location='file:' + artifact_path + '/' + exp_id)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)


def mlflow_linearize(dictionary):
    """
    Linearize a nested dictionary concatenating keys in order to allow mlflow parameters recording
    :param dictionary: nested dict
    :return: one level dict
    """
    exps = {}
    for key, value in dictionary.items():
        if isinstance(value, collections.abc.Mapping):
            exps = {**exps,
                    **{key + '.' + lin_key: lin_value for lin_key, lin_value in mlflow_linearize(value).items()}}
        else:
            exps[key] = value
    return exps


def get_experiment_loggers(exp_name, destination_folder, mlflow_logger):
    """
    Get the two loggers required for the Experimenter
    :param exp_name: unique experiment name
    :param destination_folder: folder where to save the log
    :return: logger, callback_logger
    """
    logger = mlflow_logger
    file_handler = FlushFileHandler(os.path.join(destination_folder, 'log.txt'))
    formatter = logging.Formatter('%(asctime)s %(message)s', '[%H:%M:%S]')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    callback_logger = logging.getLogger(exp_name + '_callback')
    callback_logger.addHandler(file_handler)
    callback_logger.setLevel(logging.INFO)
    return logger, callback_logger
