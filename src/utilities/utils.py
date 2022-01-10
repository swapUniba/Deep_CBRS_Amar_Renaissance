import logging
import os
import time
from functools import reduce
from itertools import groupby, product
from logging import FileHandler, LogRecord

import mlflow
import pandas as pd
import csv
import numpy as np
import json
import tensorflow as tf
import collections

from tensorflow import keras
from keras.utils.layer_utils import count_params


def read_bert_embeddings(filename_users, filename_items):
    # reading embeddings
    user_embeddings = pd.read_json(filename_users)
    item_embeddings = pd.read_json(filename_items)
    user_embeddings = user_embeddings.sort_values(by=["ID_OpenKE"])
    item_embeddings = item_embeddings.sort_values(by=["ID_OpenKE"])
    return user_embeddings, item_embeddings


def read_bert_embedding(filename_bert):
    # reading embeddings
    bert_embeddings = pd.read_json(filename_bert)
    bert_embeddings = bert_embeddings.sort_values(by=["ID_OpenKE"])
    return bert_embeddings


def read_graph_embeddings(filename_graph):
    # reading embeddings
    with open(filename_graph) as json_file:
        embeddings = json.load(json_file)
        ent_embeddings = embeddings['ent_embeddings']
    return ent_embeddings


def read_ratings(filename):
    user = []
    item = []
    rating = []
    # reading item ids
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        # next(csv_reader)
        for row in csv_reader:
            user.append(int(row[0]))
            item.append(int(row[1]))
            rating.append(int(row[2]))
    return user, item, rating


def matching_graph_emb_id(user, item, rating, ent_embeddings):
    y = np.array(rating)
    dim_embeddings = len(ent_embeddings[0])
    dim_X_cols = 2
    dim_X_rows = len(user)
    X = np.empty(shape=(dim_X_rows, dim_X_cols, dim_embeddings), dtype=np.float32)
    # matching between ids and embeddings
    i = 0
    while i < dim_X_rows:
        user_id = user[i]
        item_id = item[i]
        X[i][0] = ent_embeddings[user_id]
        X[i][1] = ent_embeddings[item_id]
        i = i + 1
    return X, y, dim_embeddings


def matching_bert_emb_id(user, item, rating, user_embeddings, item_embeddings):
    y = np.array(rating)
    dim_embeddings = len(item_embeddings['embedding'][0])
    dim_X_cols = 2
    dim_X_rows = len(user)
    X = np.empty(shape=(dim_X_rows, dim_X_cols, dim_embeddings), dtype=np.float32)
    # matching between ids and embeddings
    i = 0
    while i < dim_X_rows:
        user_id = user[i]
        item_id = item[i]
        # print(str(user_id)+"_"+str(item_id))
        user_embedding = \
            (np.array(user_embeddings.loc[(user_embeddings["ID_OpenKE"] == user_id), "profile_embedding"]))[0]
        item_embedding = (np.array(item_embeddings.loc[(item_embeddings["ID_OpenKE"] == item_id), "embedding"]))[0]
        X[i][0] = user_embedding
        X[i][1] = item_embedding
        i = i + 1

    return X, y, dim_embeddings


def matching_userBert_itemGraph(user, item, rating, user_embeddings, item_embeddings):
    y = np.array(rating)
    dim_embeddings = len(user_embeddings['profile_embedding'][0])
    dim_X_cols = 2
    dim_X_rows = len(user)
    X = np.empty(shape=(dim_X_rows, dim_X_cols, dim_embeddings), dtype=np.float32)
    # matching between ids and embeddings
    i = 0
    while i < dim_X_rows:
        user_id = user[i]
        item_id = item[i]
        user_embedding = \
            (np.array(user_embeddings.loc[(user_embeddings["ID_OpenKE"] == user_id), "profile_embedding"]))[0]
        X[i][0] = user_embedding
        X[i][1] = item_embeddings[item_id]
        i = i + 1

    return X, y, dim_embeddings


def matching_userGraph_itemBert(user, item, rating, user_embeddings, item_embeddings):
    y = np.array(rating)
    dim_embeddings = len(item_embeddings['embedding'][0])
    dim_X_cols = 2
    dim_X_rows = len(user)
    X = np.empty(shape=(dim_X_rows, dim_X_cols, dim_embeddings), dtype=np.float32)
    # matching between ids and embeddings
    i = 0
    while i < dim_X_rows:
        user_id = user[i]
        item_id = item[i]
        item_embedding = (np.array(item_embeddings.loc[(item_embeddings["ID_OpenKE"] == item_id), "embedding"]))[0]
        X[i][0] = user_embeddings[user_id]
        X[i][1] = item_embedding
        i = i + 1

    return X, y, dim_embeddings


def matching_Bert_Graph_conf(user, item, rating, graph_embeddings, user_bert_embeddings, item_bert_embeddings):
    y = np.array(rating)
    dim_embeddings = len(item_bert_embeddings['embedding'][0])
    dim_X_cols = 4
    dim_X_rows = len(user)
    X = np.empty(shape=(dim_X_rows, dim_X_cols, dim_embeddings), dtype=np.float32)
    # matching between ids and embeddings (graph and bert)
    i = 0
    while i < dim_X_rows:
        user_id = user[i]
        item_id = item[i]
        user_bert_embedding = \
            (np.array(user_bert_embeddings.loc[(user_bert_embeddings["ID_OpenKE"] == user_id), "profile_embedding"]))[0]
        item_bert_embedding = \
            (np.array(item_bert_embeddings.loc[(item_bert_embeddings["ID_OpenKE"] == item_id), "embedding"]))[0]
        X[i][0] = graph_embeddings[user_id]  # user graph
        X[i][1] = graph_embeddings[item_id]  # item graph
        X[i][2] = user_bert_embedding  # user bert
        X[i][3] = item_bert_embedding  # item bert
        i = i + 1

    return X, y, dim_embeddings


def matching_Bert_Graph(user, item, rating, graph_embeddings, user_bert_embeddings, item_bert_embeddings):
    y = np.array(rating)
    dim_embeddings_bert = len(item_bert_embeddings['embedding'][0])
    dim_embeddings_graph = len(graph_embeddings[0])
    dim_X_cols = 2
    dim_X_rows = len(user)
    X_graph = np.empty(shape=(dim_X_rows, dim_X_cols, dim_embeddings_graph), dtype=np.float32)
    X_bert = np.empty(shape=(dim_X_rows, dim_X_cols, dim_embeddings_bert), dtype=np.float32)
    # matching between ids and embeddings (graph and bert)
    i = 0
    while i < dim_X_rows:
        user_id = user[i]
        item_id = item[i]
        user_bert_embedding = \
            (np.array(user_bert_embeddings.loc[(user_bert_embeddings["ID_OpenKE"] == user_id), "profile_embedding"]))[0]
        item_bert_embedding = \
            (np.array(item_bert_embeddings.loc[(item_bert_embeddings["ID_OpenKE"] == item_id), "embedding"]))[0]
        X_graph[i][0] = graph_embeddings[user_id]  # user graph
        X_graph[i][1] = graph_embeddings[item_id]  # item graph
        X_bert[i][0] = user_bert_embedding  # user bert
        X_bert[i][1] = item_bert_embedding  # item bert
        i = i + 1

    return X_graph, X_bert, dim_embeddings_graph, dim_embeddings_bert, y


def top_scores(predictions, n):
    top_n_scores = pd.DataFrame()
    for u in list(set(predictions['users'])):
        p = predictions.loc[predictions['users'] == u]
        top_n_scores = top_n_scores.append(p.head(n))
    return top_n_scores


class LogCallback(keras.callbacks.Callback):
    def __init__(self, log, frequency):
        """

        :param log: log object
        :param writer: Summary Writer
        :param frequency: frequency of logging batches (too much frequency will slow down the process)
        """
        super().__init__()
        self.log = log
        self.log_frequency = frequency
        self.train_start = None

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        self.train_start = time.perf_counter()
        self.log.info("Starting training - got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        # tf.summary.scalar('train_time', time.perf_counter() - self.train_start, step=0)
        self.log.info("End training")

    def on_epoch_begin(self, epoch, logs=None):
        self.log.info('Epoch: {} '.format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        msg = reduce(lambda a, b: a + b, ["{}: {},\t".format(key, value) for key, value in logs.items()])
        self.log.info("End epoch {} of training - {}".format(epoch, msg))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        self.log.info("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        self.log.info("Stop testing; got log keys: {}".format(keys))

    # def on_train_batch_end(self, batch, logs=None):
    #     if batch % self.log_frequency == 0:
    #         msg = reduce(lambda a, b: a + b, ["{}: {},\t".format(key, value) for key, value in logs.items()])
    #         self.log.info("Batch {} \t - {}".format(batch, msg))

    def on_test_batch_end(self, batch, logs=None):
        if batch % self.log_frequency == 0:
            msg = reduce(lambda a, b: a + b, ["{}: {},\t".format(key, value) for key, value in logs.items()])
            self.log.info("Batch {} \t - {}".format(batch, msg))


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


def get_experiment_loggers(exp_name, destination_folder, mlflow_logger):
    """
    Get the two loggers required for the Experimenter
    :param exp_name: unique experiment name
    :param destination_folder: folder where to save the log
    :param mlflow_logger: add FileHandler to mlflow logger
    :return: logger, callback_logger
    """
    file_handler = FlushFileHandler(os.path.join(destination_folder, 'log.txt'))
    formatter = logging.Formatter('%(asctime)s %(message)s', '[%H:%M:%S]')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = mlflow_logger
    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    callback_logger = logging.getLogger(exp_name + '_callback')
    callback_logger.addHandler(file_handler)
    callback_logger.setLevel(logging.INFO)
    return logger, callback_logger


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
