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
        elif isinstance(value, list):
            exps.append((key, value))
        else:
          raise ValueError("Only dict or lists!!!")
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


def setup_mlflow(exp_name, mlflow_path):
    """
    """
    mlflow.tensorflow.autolog()
    os.makedirs(mlflow_path, exist_ok=True)
    os.makedirs(os.path.join(mlflow_path, '.trash'), exist_ok=True)

    experiment = mlflow.get_experiment_by_name(exp_name)
    if not experiment:
        exps = os.listdir(mlflow_path)
        exps.pop(exps.index('.trash'))
        if len(exps) == 0:
            exp_id = '0'
        else:
            exp_id = str(max([int(exp) for exp in exps]) + 1)
        exp_path = mlflow_path + '/' + exp_id
        experiment_id = mlflow.create_experiment(
            exp_name, artifact_location='file:' + exp_path)
    else:
        experiment_id = experiment.experiment_id
        exp_path = experiment.artifact_location.split(':')[1]
    mlflow.set_experiment(experiment_id=experiment_id)
    return exp_path


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


def get_experiment_logger(destination_folder):
    """
    Get the logger required for the Experimenter.

    :param destination_folder: folder where to save the log
    :return: logger
    """
    # Instantiate the formatted and force-flush file handler
    formatter = logging.Formatter('%(asctime)s %(message)s', '[%H:%M:%S]')
    file_handler = FlushFileHandler(os.path.join(destination_folder, 'log.txt'))
    file_handler.setFormatter(formatter)

    # Instantiate the logger
    logger = logging.getLogger('callback')
    for handler in logger.handlers:  # Delete all the current handlers
        logger.removeHandler(handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Do not write also on stdout (i.e. don't propagate to upper-level logger)
    return logger


def twgnn_debugger(model, inputs):
    import tensorflow as tf
    from spektral.layers import ops
    from tensorflow.keras import backend as K

    with tf.GradientTape(persistent=True) as g:
        g.watch(inputs)
        u, i = inputs

        users = model.gnn.way_one_gnn_layers(None)

        item_gnn = model.gnn.way_two_gnn_layers

        x = item_gnn.embeddings
        hs = [x]
        x = item_gnn.seq_layers[0]([x, item_gnn.adj_matrix])
        if item_gnn.dropout is not None:
            x = item_gnn.dropout(x)
        hs.append(x)

        # GAT conv 2 ###########################
        a = item_gnn.adj_matrix
        gat_conv2 = item_gnn.seq_layers[1]
        gat_conv2.build([x.shape, a.shape])

        kernel = tf.reshape(gat_conv2.kernel, (-1, gat_conv2.attn_heads * gat_conv2.channels))
        attn_kernel_self = ops.transpose(gat_conv2.attn_kernel_self, (2, 1, 0))
        attn_kernel_neighs = ops.transpose(gat_conv2.attn_kernel_neighs, (2, 1, 0))

        # Prepare message-passing
        indices = a.indices
        N = tf.shape(x, out_type=indices.dtype)[-2]
        if gat_conv2.add_self_loops:
            indices = ops.add_self_loops_indices(indices, N)
        targets, sources = indices[:, 1], indices[:, 0]

        # Update node features
        x = K.dot(x, kernel)
        x = tf.reshape(x, (-1, gat_conv2.attn_heads, gat_conv2.channels))

        # Compute attention
        attn_for_self = tf.reduce_sum(x * attn_kernel_self, -1)
        attn_for_self = tf.gather(attn_for_self, targets)
        attn_for_neighs = tf.reduce_sum(x * attn_kernel_neighs, -1)
        attn_for_neighs = tf.gather(attn_for_neighs, sources)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = ops.unsorted_segment_softmax(attn_coef, targets, N)
        attn_coef = gat_conv2.dropout(attn_coef)
        attn_coef = attn_coef[..., None]

        # Update representation
        call_single_output = attn_coef * tf.gather(x, sources)
        call_single_output = tf.math.unsorted_segment_sum(call_single_output, targets, N)
        gat_output = call_single_output

        shape = tf.concat(
            (tf.shape(gat_output)[:-2], [gat_conv2.attn_heads * gat_conv2.channels]), axis=0
        )
        output = tf.reshape(gat_output, shape)

        gat_output += gat_conv2.bias
        gat_output = gat_conv2.activation(output)

        ################
        x = gat_output
        if item_gnn.dropout is not None:
            x = item_gnn.dropout(x)
        hs.append(x)
        # Reduce the outputs of each GCN layer
        items = item_gnn.reduce(hs)

        embeddings_one = tf.concat([
            tf.slice(users, begin=[0, 0], size=[model.gnn.n_users, users.shape[1]]),
            tf.slice(items, begin=[0, 0], size=[model.gnn.n_items, items.shape[1]]),
        ], axis=0)
        embeddings = model.gnn.step_two_gnn_layers(embeddings_one)

        u = tf.nn.embedding_lookup(embeddings, u)
        i = tf.nn.embedding_lookup(embeddings, i)
        output = model.rs([u, i])

    gradients1 = g.gradient(output, inputs)
    gradients2 = g.gradient(embeddings, inputs)
    grad_one = g.gradient(embeddings, embeddings_one)
    grad_emb = [tf.reduce_mean(grad).numpy() for grad in
                g.gradient(output, model.gnn.step_two_gnn_layers.trainable_weights)]
    grad1 = [tf.reduce_mean(grad).numpy() for grad in
             g.gradient(embeddings_one, model.gnn.way_one_gnn_layers.trainable_weights)]
    # grad2 = [tf.reduce_mean(grad).numpy() for grad in g.gradient(embeddings_one, model.gnn.way_two_gnn_layers.trainable_weights)]
    # gradconv3 = [tf.reduce_mean(grad).numpy() for grad in g.gradient(embeddings_one, model.gnn.way_two_gnn_layers.seq_layers[1].trainable_weights)]