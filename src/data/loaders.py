import json

import pandas as pd
import numpy as np

from scipy import sparse

from data.datasets import UserItemEmbeddings, HybridUserItemEmbeddings, UserItemGraph, UserItemGraphEmbeddings


def build_adjacency_matrix(
        bi_ratings,
        users,
        items,
        props_triples=None,
        props=None,
        binary_adjacency=False,
        sparse_adjacency=True,
        symmetric_adjacency=True
):
    """
    :param bi_ratings: The bipartite ratings as a matrix associating to users and items a 0-1 rating.
    :param users: A sequence of users IDs.
    :param items: A sequence of items IDs.
    :param props_triples: The knowledge graph triples of items and properties. It can be None.
    :param props: A sequence of properties IDs. It can be None.
    :param binary_adjacency: Used only if return_adjacency is True and sparse_adjacency is True. Whether to consider
                             both positive and negative ratings, hence returning an adjacency matrix with 0 and 1.
    :param sparse_adjacency: User only if binary_adjacency is False. Whether to return the adjacency matrix as a sparse
                             matrix instead of dense.
    :param symmetric_adjacency: Whether to return a symmetric adjacency matrix.
    :return: The adjacency matrix.
    """
    # Compute the dimensions of the adjacency matrix
    adj_size = len(users) + len(items)
    if props is not None:
        adj_size += len(props)

    # Compute the adjacency matrix
    if binary_adjacency:
        if not sparse_adjacency:
            raise NotImplementedError("A multi-relational adjacency matrix must be sparse")
        coo_data = bi_ratings[:, 2]
        coo_rows, coo_cols = bi_ratings[:, 0], bi_ratings[:, 1]
        if symmetric_adjacency:
            coo_data = np.concatenate([coo_data, coo_data])
            coo_rows, coo_cols = np.concatenate([coo_rows, coo_cols]), np.concatenate([coo_cols, coo_rows])
        adj_matrix = sparse.coo_matrix(
            (coo_data, (coo_rows, coo_cols)),
            shape=[adj_size, adj_size], dtype=np.float32
        )
    else:
        pos_idx = bi_ratings[:, 2] == 1
        adj_matrix = sparse.coo_matrix(
            (bi_ratings[pos_idx, 2], (bi_ratings[pos_idx, 0], bi_ratings[pos_idx, 1])),
            shape=[adj_size, adj_size], dtype=np.float32
        )
        if symmetric_adjacency:
            adj_matrix += adj_matrix.T

    # Convert to dense matrix
    if not sparse_adjacency:
        adj_matrix = adj_matrix.todense()

    return adj_matrix


def load_train_test_ratings(
        train_filepath,
        test_filepath,
        props_filepath=None,
        sep='\t',
        return_adjacency=False,
        binary_adjacency=False,
        sparse_adjacency=True,
        symmetric_adjacency=True
):
    """
    Load train and test ratings. Note that the user and item IDs are converted to sequential numbers.

    :param train_filepath: The training ratings CSV or TSV filepath.
    :param test_filepath: The test ratings CSV or TSV filepath.
    :param props_filepath: The properties triples CSV or TSV filepath. It can be None, and it is used only if
                           return_adjacency is True.
    :param sep: The separator to use for CSV or TSV files.
    :param return_adjacency: Whether to also return the adjacency matrix.
    :param binary_adjacency: Used only if return_adjacency is True and sparse_adjacency is True. Whether to consider
                             both positive and negative ratings, hence returning an adjacency matrix with 0 and 1.
    :param sparse_adjacency: User only if binary_adjacency is False. Whether to return the adjacency matrix as a sparse
                             matrix instead of dense.
    :param symmetric_adjacency: Whether to return a symmetric adjacency matrix.
    :return: The training and test ratings as an array of User-Item-Rating where IDs are made sequential.
             Moreover, it returns the users and items original unique IDs. Additionally, it also returns the training
             interactions adjacency matrix (assuming un-directed arcs).
    """
    # Load the ratings arrays
    train_ratings = pd.read_csv(train_filepath, sep=sep, header=None).to_numpy()
    test_ratings = pd.read_csv(test_filepath, sep=sep, header=None).to_numpy()

    # Convert users and items ids to indices (i.e. sequential)
    users, users_indexes = np.unique(train_ratings[:, 0], return_inverse=True)
    items, items_indexes = np.unique(train_ratings[:, 1], return_inverse=True)
    items_indexes += len(users)
    train_ratings = np.stack([users_indexes, items_indexes, train_ratings[:, 2]], axis=1)

    # Do the same for the test ratings, by using the same users and items of the train ratings
    users_indexes = np.argwhere(test_ratings[:, [0]] == users)[:, 1]
    items_indexes = np.argwhere(test_ratings[:, [1]] == items)[:, 1]
    items_indexes += len(users)
    test_ratings = np.stack([users_indexes, items_indexes, test_ratings[:, 2]], axis=1)

    if not return_adjacency:
        return (train_ratings, test_ratings), (users, items)

    # Load the properties, if specified
    if props_filepath is not None:
        props_triples = pd.read_csv(props_filepath, sep=sep, header=None).to_numpy()
        items_indexes = np.argwhere(props_triples[:, [0]] == items)[:, 1]
        props, props_indexes = np.unique(props_triples[:, 1], return_inverse=True)
        rels, rels_indexes = np.unique(props_triples[:, 2], return_inverse=True)
        items_indexes += len(users)
        props_indexes += len(users) + len(items)
        rels_indexes += 2  # We already have 0-1 ratings for users and items
        props_triples = np.stack([items_indexes, props_indexes, rels_indexes], axis=1)
    else:
        props = None
        props_triples = None

    # Build the adjacency matrix
    adj_matrix = build_adjacency_matrix(
        train_ratings, users, items,
        props_triples=props_triples, props=props,
        binary_adjacency=binary_adjacency,
        sparse_adjacency=sparse_adjacency,
        symmetric_adjacency=symmetric_adjacency
    )

    return (train_ratings, test_ratings), (users, items), adj_matrix


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
    return np.concatenate([user_embeddings, item_embeddings], axis=0)


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
    return np.concatenate([user_embeddings, item_embeddings], axis=0)


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
                                sep=sep,
                                return_adjacency=False)

    graph_embeddings = load_graph_user_item_embeddings(graph_filepath, users, items)

    data_train = UserItemEmbeddings(
        train_ratings, users, items, graph_embeddings,
        batch_size=train_batch_size, shuffle=shuffle
    )
    data_test = UserItemEmbeddings(
        test_ratings, users, items, graph_embeddings,
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
                                sep=sep,
                                return_adjacency=False)

    bert_embeddings = load_bert_user_item_embeddings(bert_user_filepath, bert_item_filepath, users, items)

    data_train = UserItemEmbeddings(
        train_ratings, users, items, bert_embeddings,
        batch_size=train_batch_size, shuffle=shuffle
    )
    data_test = UserItemEmbeddings(
        test_ratings, users, items, bert_embeddings,
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
                                sep=sep,
                                return_adjacency=False)

    graph_embeddings = load_graph_user_item_embeddings(graph_filepath, users, items)
    bert_embeddings = load_bert_user_item_embeddings(bert_user_filepath, bert_item_filepath, users, items)

    data_train = HybridUserItemEmbeddings(
        train_ratings, users, items, graph_embeddings, bert_embeddings,
        batch_size=train_batch_size, shuffle=shuffle
    )
    data_test = HybridUserItemEmbeddings(
        test_ratings, users, items, graph_embeddings, bert_embeddings,
        batch_size=test_batch_size, shuffle=False
    )
    return data_train, data_test


def load_user_item_graph(
        train_ratings_filepath,
        test_ratings_filepath,
        sep='\t',
        binary_adjacency=False,
        sparse_adjacency=True,
        symmetric_adjacency=True,
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
    :param symmetric_adjacency: Whether to return a symmetric adjacency matrix.
    :param shuffle: Tells if shuffle the training dataset.
    :param train_batch_size: batch_size used in training phase.
    :param test_batch_size: batch_size used in test phase.
    :return: The training and test ratings data sequence for GNN-based models.
    """
    (train_ratings, test_ratings), (users, items), adj_matrix = \
        load_train_test_ratings(train_ratings_filepath,
                                test_ratings_filepath,
                                sep=sep,
                                return_adjacency=True,
                                binary_adjacency=binary_adjacency,
                                sparse_adjacency=sparse_adjacency,
                                symmetric_adjacency=symmetric_adjacency)
    data_train = UserItemGraph(
        train_ratings, users, items, adj_matrix,
        batch_size=train_batch_size, shuffle=shuffle
    )
    data_test = UserItemGraph(
        test_ratings, users, items, adj_matrix,
        batch_size=test_batch_size, shuffle=False
    )
    return data_train, data_test


def load_user_item_graph_bert_embeddings(
        train_ratings_filepath,
        test_ratings_filepath,
        bert_user_filepath,
        bert_item_filepath,
        sep='\t',
        binary_adjacency=False,
        sparse_adjacency=True,
        symmetric_adjacency=True,
        shuffle=True,
        train_batch_size=1024,
        test_batch_size=2048
):
    """
    Load train and test ratings for GNN-based models.
    Note that the user and item IDs are converted to sequential numbers.

    :param train_ratings_filepath: The training ratings CSV or TSV filepath.
    :param test_ratings_filepath: The test ratings CSV or TSV filepath.
    :param bert_user_filepath: The filepath for User BERT embeddings.
    :param bert_item_filepath: The filepath for Item BERT embeddings.
    :param sep: The separator to use for CSV or TSV files.
    :param binary_adjacency: Used only if return_adjacency is True. Whether to consider both positive and negative
                             ratings, hence returning two adjacency matrices as an array of shape (2, n_nodes, n_nodes).
    :param sparse_adjacency: User only if binary_adjacency is False. Whether to return the adjacency matrix as a sparse
                             matrix instead of dense.
    :param symmetric_adjacency: Whether to return a symmetric adjacency matrix.
    :param shuffle: Tells if shuffle the training dataset.
    :param train_batch_size: batch_size used in training phase.
    :param test_batch_size: batch_size used in test phase.
    :return: The training and test ratings data sequence for GNN-based models.
    """
    (train_ratings, test_ratings), (users, items), adj_matrix = \
        load_train_test_ratings(train_ratings_filepath,
                                test_ratings_filepath,
                                sep=sep,
                                return_adjacency=True,
                                binary_adjacency=binary_adjacency,
                                sparse_adjacency=sparse_adjacency,
                                symmetric_adjacency=symmetric_adjacency)

    bert_embeddings = load_bert_user_item_embeddings(bert_user_filepath, bert_item_filepath, users, items)

    data_train = UserItemGraphEmbeddings(
        train_ratings, users, items, adj_matrix, bert_embeddings,
        batch_size=train_batch_size, shuffle=shuffle
    )
    data_test = UserItemGraphEmbeddings(
        test_ratings, users, items, adj_matrix, bert_embeddings,
        batch_size=test_batch_size, shuffle=False
    )
    return data_train, data_test
