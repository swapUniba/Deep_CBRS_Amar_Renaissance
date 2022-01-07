# Forse è meglio trovare un nome migliore
import json

import pandas as pd
import numpy as np

from scipy import sparse

from utilities.datasets import UserItemEmbeddings, HybridUserItemEmbeddings, UserItemGraph


def load_train_test_ratings(
        train_filepath,
        test_filepath,
        sep='\t',
        return_adjacency=False,
        binary_adjacency=False,
        sparse_adjacency=True
):
    """
    Load train and test ratings. Note that the user and item IDs are converted to sequential numbers.

    :param train_filepath: The training ratings CSV or TSV filepath.
    :param test_filepath: The test ratings CSV or TSV filepath.
    :param sep: The separator to use for CSV or TSV files.
    :param return_adjacency: Whether to also return the adjacency matrix.
    :param binary_adjacency: Used only if return_adjacency is True. Whether to consider both positive and negative
                             ratings, hence returning two adjacency matrices as an array of shape (2, n_nodes, n_nodes).
    :param sparse_adjacency: User only if binary_adjacency is False. Whether to return the adjacency matrix as a sparse
                             matrix instead of dense.
    :return: The training and test ratings as an array of User-Item-Rating where IDs are made sequential.
             Moreover, it returns the users and items original unique IDs if return_adjacency is False,
             otherwise it returns the training interactions adjacency matrix (assuming un-directed arcs).
    """
    # Load the ratings arrays
    train_ratings = pd.read_csv(train_filepath, sep=sep).to_numpy()
    test_ratings = pd.read_csv(test_filepath, sep=sep).to_numpy()

    # Convert users and items ids to indices (i.e. sequential)
    users, users_indexes = np.unique(train_ratings[:, 0], return_inverse=True)
    items, items_indexes = np.unique(train_ratings[:, 1], return_inverse=True)
    train_ratings = np.stack([users_indexes, items_indexes, train_ratings[:, 2]], axis=1)

    # Do the same for the test ratings, by using the same users and items of the train ratings
    users_indexes = np.argwhere(test_ratings[:, [0]] == users)[:, 1]
    items_indexes = np.argwhere(test_ratings[:, [1]] == items)[:, 1]
    test_ratings = np.stack([users_indexes, items_indexes, test_ratings[:, 2]], axis=1)

    if not return_adjacency:
        return (train_ratings, test_ratings), (users, items)

    # Add a constant factors to item ids
    train_ratings[:, 1] += len(users)
    test_ratings[:, 1] += len(users)

    # Compute the dimensions of the adjacency matrix
    adj_size = len(users) + len(items)

    # Compute the adjacency matrix
    pos_idx = train_ratings[:, 2] == 1
    if binary_adjacency:
        adj_matrix = np.zeros([2, adj_size, adj_size], dtype=np.float32)
        adj_matrix[0, train_ratings[pos_idx, 0], train_ratings[pos_idx, 1]] = 1.0
        adj_matrix[1, train_ratings[~pos_idx, 0], train_ratings[~pos_idx, 1]] = 1.0
        adj_matrix += np.transpose(adj_matrix, axes=[0, 2, 1])
    else:
        if sparse_adjacency:
            adj_matrix = sparse.coo_matrix(
                (train_ratings[pos_idx, 2], (train_ratings[pos_idx, 0], train_ratings[pos_idx, 1])),
                shape=[adj_size, adj_size], dtype=np.float32
            )
            adj_matrix += adj_matrix.T
        else:
            adj_matrix = np.zeros([adj_size, adj_size], dtype=np.float32)
            adj_matrix[train_ratings[pos_idx, 0], train_ratings[pos_idx, 1]] = 1.0
            adj_matrix += adj_matrix.T

    return (train_ratings, test_ratings), adj_matrix


def json_load_graph_embeddings(filepath):
    with open(filepath) as fp:
        embeddings = json.load(fp)
    return embeddings['ent_embeddings']


def json_load_bert_embeddings(filepath):
    embeddings = pd.read_json(filepath)
    return embeddings.sort_values(by=['ID_OpenKE'])


def load_graph_user_item_embeddings(filepath, users, items):
    graph_embeddings = np.array(json_load_graph_embeddings(filepath), dtype=np.float32)
    user_embeddings = graph_embeddings[users]
    item_embeddings = graph_embeddings[items]
    return user_embeddings, item_embeddings


def load_bert_user_item_embeddings(user_filepath, item_filepath, users, items):
    user_embeddings, item_embeddings = dict(), dict()
    df_users = json_load_bert_embeddings(user_filepath)
    df_items = json_load_bert_embeddings(item_filepath)
    for _, user in df_users.iterrows():
        user_id = user['ID_OpenKE']
        user_embeddings[user_id] = np.array(user['profile_embedding'], dtype=np.float32)
    for _, item in df_items.iterrows():
        item_id = item['ID_OpenKE']
        item_embeddings[item_id] = np.array(item['embedding'], dtype=np.float32)
    user_embeddings = np.stack([user_embeddings[u] for u in users])
    item_embeddings = np.stack([item_embeddings[i] for i in items])
    return user_embeddings, item_embeddings


# Train, test load functions

def load_graph_embeddings(
        train_ratings_filepath,
        test_ratings_filepath,
        graph_filepath,
        sep='\t',
        shuffle=True,
        train_batch_size=1024,
        test_batch_size=2048
):
    """
    Load train and test ratings datasets consisting of Graph embeddings.

    :param train_ratings_filepath: The training ratings CSV or TSV filepath.
    :param test_ratings_filepath: The test ratings CSV or TSV filepath.
    :param graph_filepath: The filepath for Graph embeddings.
    :param sep: The separator to use for CSV or TSV files.
    :param shuffle: Tells if shuffle the training dataset.
    :param train_batch_size: batch_size used in training phase.
    :param test_batch_size: batch_size used in test phase.
    :return: The training and test ratings data sequence for graph embeddings RS models.
    """
    (train_ratings, test_ratings), (users, items) = \
        load_train_test_ratings(train_ratings_filepath,
                                test_ratings_filepath,
                                sep,
                                return_adjacency=False)

    graph_user_embeddings, graph_item_embeddings = \
        load_graph_user_item_embeddings(graph_filepath, users, items)

    data_train = UserItemEmbeddings(
        train_ratings, graph_user_embeddings, graph_item_embeddings,
        batch_size=train_batch_size, shuffle=shuffle
    )
    data_test = UserItemEmbeddings(
        test_ratings, graph_user_embeddings, graph_item_embeddings,
        batch_size=test_batch_size, shuffle=False
    )
    return data_train, data_test


def load_bert_embeddings(
        train_ratings_filepath,
        test_ratings_filepath,
        bert_user_filepath,
        bert_item_filepath,
        sep='\t',
        shuffle=True,
        train_batch_size=1024,
        test_batch_size=2048
):
    """
    Load train and test ratings datasets consisting of BERT embeddings.

    :param train_ratings_filepath: The training ratings CSV or TSV filepath.
    :param test_ratings_filepath: The test ratings CSV or TSV filepath.
    :param bert_user_filepath: The filepath for User BERT embeddings.
    :param bert_item_filepath: The filepath for Item BERT embeddings.
    :param sep: The separator to use for CSV or TSV files.
    :param shuffle: Tells if shuffle the training dataset.
    :param train_batch_size: batch_size used in training phase.
    :param test_batch_size: batch_size used in test phase.
    :return: The training and test ratings data sequence for graph embeddings RS models.
    """
    (train_ratings, test_ratings), (users, items) = \
        load_train_test_ratings(train_ratings_filepath,
                                test_ratings_filepath,
                                sep,
                                return_adjacency=False)

    bert_user_embeddings, bert_item_embeddings = \
        load_bert_user_item_embeddings(bert_user_filepath, bert_item_filepath, users, items)

    data_train = UserItemEmbeddings(
        train_ratings, bert_user_embeddings, bert_item_embeddings,
        batch_size=train_batch_size, shuffle=shuffle
    )
    data_test = UserItemEmbeddings(
        test_ratings, bert_user_embeddings, bert_item_embeddings,
        batch_size=test_batch_size, shuffle=False
    )
    return data_train, data_test


def load_hybrid_embeddings(
        train_ratings_filepath,
        test_ratings_filepath,
        graph_filepath,
        bert_user_filepath,
        bert_item_filepath,
        sep='\t',
        shuffle=True,
        train_batch_size=1024,
        test_batch_size=2048
):
    """
    Load train and test ratings datasets consisting of BERT+Graph embeddings.

    :param train_ratings_filepath: The training ratings CSV or TSV filepath.
    :param test_ratings_filepath: The test ratings CSV or TSV filepath.
    :param graph_filepath: The filepath for Graph embeddings.
    :param bert_user_filepath: The filepath for User BERT embeddings.
    :param bert_item_filepath: The filepath for Item BERT embeddings.
    :param sep: The separator to use for CSV or TSV files.
    :param shuffle: Tells if shuffle the training dataset.
    :param train_batch_size: batch_size used in training phase.
    :param test_batch_size: batch_size used in test phase.
    :return: The training and test ratings data sequence for hybrid CBRS models.
    """
    (train_ratings, test_ratings), (users, items) = \
        load_train_test_ratings(train_ratings_filepath,
                                test_ratings_filepath,
                                sep,
                                return_adjacency=False)

    graph_user_embeddings, graph_item_embeddings = \
        load_graph_user_item_embeddings(graph_filepath, users, items)
    bert_user_embeddings, bert_item_embeddings = \
        load_bert_user_item_embeddings(bert_user_filepath, bert_item_filepath, users, items)

    data_train = HybridUserItemEmbeddings(
        train_ratings, graph_user_embeddings, graph_item_embeddings,
        bert_user_embeddings, bert_item_embeddings, batch_size=train_batch_size, shuffle=shuffle
    )
    data_test = HybridUserItemEmbeddings(
        test_ratings, graph_user_embeddings, graph_item_embeddings,
        bert_user_embeddings, bert_item_embeddings, batch_size=test_batch_size, shuffle=False
    )
    return data_train, data_test


def load_user_item_graph(
        train_ratings_filepath,
        test_ratings_filepath,
        sep='\t',
        binary_adjacency=False,
        sparse_adjacency=True,
        shuffle=True,
        train_batch_size=1024,
        test_batch_size=2048
):
    """
    Load train and test ratings for GNN-based models.
    Note that the user and item IDs are converted to sequential numbers.

    :param train_ratings_filepath: The training ratings CSV or TSV filepath.
    :param test_ratings_filepath: The test ratings CSV or TSV filepath.
    :param sep: The separator to use for CSV or TSV files.
    :param binary_adjacency: Used only if return_adjacency is True. Whether to consider both positive and negative
                             ratings, hence returning two adjacency matrices as an array of shape (2, n_nodes, n_nodes).
    :param sparse_adjacency: User only if binary_adjacency is False. Whether to return the adjacency matrix as a sparse
                             matrix instead of dense.
    :param shuffle: Tells if shuffle the training dataset.
    :param train_batch_size: batch_size used in training phase.
    :param test_batch_size: batch_size used in test phase.
    :return: The training and test ratings data sequence for GNN-based models.
    """
    (train_ratings, test_ratings), adj_matrix = \
        load_train_test_ratings(train_ratings_filepath,
                                test_ratings_filepath,
                                sep,
                                return_adjacency=True,
                                binary_adjacency=binary_adjacency,
                                sparse_adjacency=sparse_adjacency)
    data_train = UserItemGraph(
        train_ratings, adj_matrix,
        batch_size=train_batch_size, shuffle=shuffle
    )
    data_test = UserItemGraph(
        test_ratings, adj_matrix,
        batch_size=test_batch_size, shuffle=False
    )
    return data_train, data_test
