import json

import pandas as pd
import numpy as np

from scipy import sparse

from utilities.math import symmetrize_matrix
from data.datasets import UserItemEmbeddings, HybridUserItemEmbeddings
from data.datasets import UserItemGraph, UserItemGraphEmbeddings, UserItemGraphPosNegSample


def build_adjacency_matrix(
        bi_ratings,
        users,
        items,
        props_triples=None,
        props=None,
        type_adjacency='unary',
        sparse_adjacency=True,
        symmetric_adjacency=True
):
    """
    :param bi_ratings: The bipartite ratings as a matrix associating to users and items a 0-1 rating.
    :param users: A sequence of users IDs.
    :param items: A sequence of items IDs.
    :param props_triples: The knowledge graph triples of items and properties. It can be None.
    :param props: A sequence of properties IDs. It can be None.
    :param type_adjacency: Used only if return_adjacency is True and sparse_adjacency is True. It can be either
                           'unary' for 1-only ratings, 'binary' for 0/1-only ratings and 'unary-kg' if you need both the
                           unary matrix and the KG unary graph. In the latter case it requires props and props_triples.
    :param sparse_adjacency: Whether to return the adjacency matrix as a sparse matrix instead of dense.
    :param symmetric_adjacency: Whether to return a symmetric adjacency matrix.
    :return: The adjacency matrix.
    """
    # Unary adjacency matrix (only positive ratings)
    if type_adjacency == 'unary':
        # Compute the dimensions of the adjacency matrix
        adj_size = len(users) + len(items)

        # Instantiate the sparse adjacency matrix
        pos_idx = bi_ratings[:, 2] == 1
        adj_matrix = sparse.coo_matrix(
            (bi_ratings[pos_idx, 2], (bi_ratings[pos_idx, 0], bi_ratings[pos_idx, 1])),
            shape=[adj_size, adj_size], dtype=np.float32
        )

        # Make the matrix symmetric
        if symmetric_adjacency:
            adj_matrix = symmetrize_matrix(adj_matrix)

        # Convert to dense matrix, if specified
        if not sparse_adjacency:
            adj_matrix = adj_matrix.todense()
        return adj_matrix

    # Binary adjacency matrix (both positive and negative ratings)
    if type_adjacency == 'binary':
        # Compute the dimensions of the adjacency matrix
        adj_size = len(users) + len(items)

        # Set the data of the matrix
        coo_data = bi_ratings[:, 2]
        coo_rows, coo_cols = bi_ratings[:, 0], bi_ratings[:, 1]

        # Instantiate the sparse adjacency matrix
        adj_matrix = sparse.coo_matrix(
            (coo_data, (coo_rows, coo_cols)),
            shape=[adj_size, adj_size], dtype=np.float32
        )

        # Make the matrix symmetric
        if symmetric_adjacency:
            adj_matrix = symmetrize_matrix(adj_matrix)

        # Convert to dense matrix, if specified
        if not sparse_adjacency:
            adj_matrix = adj_matrix.todense()
        return adj_matrix

    # Unary KG adjacency matrices (return both unary adjacency matrix and KG adjacency matrix)
    if type_adjacency == 'unary-kg':
        if props is None or props_triples is None:
            raise ValueError("KG adjacency matrix requires properties info")

        # Compute the dimensions of the adjacency matrices
        adj_bi_size = len(users) + len(items)
        adj_kg_size = len(items) + len(props)

        # Instantiate the sparse adjacency matrices
        pos_idx = bi_ratings[:, 2] == 1
        coo_data = bi_ratings[pos_idx, 2]
        coo_rows, coo_cols = bi_ratings[pos_idx, 0], bi_ratings[pos_idx, 1]
        adj_bi_matrix = sparse.coo_matrix(
            (coo_data, (coo_rows, coo_cols)),
            shape=[adj_bi_size, adj_bi_size], dtype=np.float32
        )
        coo_data = props_triples[:, 2]
        coo_rows, coo_cols = props_triples[:, 0], props_triples[:, 1]
        adj_kg_matrix = sparse.coo_matrix(
            (coo_data, (coo_rows, coo_cols)),
            shape=[adj_kg_size, adj_kg_size], dtype=np.float32
        )

        # Make the matrices symmetric
        if symmetric_adjacency:
            adj_bi_matrix = symmetrize_matrix(adj_bi_matrix)
            adj_kg_matrix = symmetrize_matrix(adj_kg_matrix)

        # Convert to dense matrices, if specified
        if not sparse_adjacency:
            adj_bi_matrix = adj_bi_matrix.todense()
            adj_kg_matrix = adj_kg_matrix.todense()
        return adj_bi_matrix, adj_kg_matrix

    raise ValueError("Unknown adjacency matrix type named {}".format(type_adjacency))


def load_train_test_ratings(
        train_filepath,
        test_filepath,
        props_filepath=None,
        sep='\t',
        return_adjacency=False,
        type_adjacency='unary',
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
    :param type_adjacency: Used only if return_adjacency is True and sparse_adjacency is True. It can be either
                           'unary' for 1-only ratings, 'binary' for 0/1-only ratings and 'unary-kg' if you need both the
                           unary matrix and the KG unary graph. In the latter case it requires props and props_triples.
    :param sparse_adjacency: Whether to return the adjacency matrix as a sparse matrix instead of dense.
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
    if type_adjacency == 'unary-kg' and props_filepath is not None:
        props_triples = pd.read_csv(props_filepath, sep=sep, header=None).to_numpy()
        items_indexes = np.argwhere(props_triples[:, [0]] == items)[:, 1]
        props, props_indexes = np.unique(props_triples[:, 1], return_inverse=True)
        props_indexes += len(items)
        ones = np.ones(len(props_indexes), dtype=props_triples.dtype)
        props_triples = np.stack([items_indexes, props_indexes, ones], axis=1)
    else:
        props = None
        props_triples = None

    # Build the adjacency matrix
    adj_matrix = build_adjacency_matrix(
        train_ratings, users, items,
        props_triples=props_triples, props=props,
        type_adjacency=type_adjacency,
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
        props_triples_filepath=None,
        sep='\t',
        type_adjacency='unary',
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
    :param props_triples_filepath: The properties triples CSV or TSV filepath. It can be None, and it is used only if
                                   return_adjacency is True.
    :param sep: The separator to use for CSV or TSV files.
    :param type_adjacency: Used only if return_adjacency is True and sparse_adjacency is True. It can be either
                           'unary' for 1-only ratings, 'binary' for 0/1-only ratings and 'unary-kg' if you need both the
                           unary matrix and the KG unary graph. In the latter case it requires props and props_triples.
    :param sparse_adjacency: Whether to return the adjacency matrix as a sparse matrix instead of dense.
    :param symmetric_adjacency: Whether to return a symmetric adjacency matrix.
    :param shuffle: Tells if shuffle the training dataset.
    :param train_batch_size: batch_size used in training phase.
    :param test_batch_size: batch_size used in test phase.
    :return: The training and test ratings data sequence for GNN-based models.
    """
    (train_ratings, test_ratings), (users, items), adj_matrix = \
        load_train_test_ratings(train_ratings_filepath,
                                test_ratings_filepath,
                                props_triples_filepath,
                                sep=sep,
                                return_adjacency=True,
                                type_adjacency=type_adjacency,
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


def load_user_item_graph_sample(
        train_ratings_filepath,
        test_ratings_filepath,
        props_triples_filepath=None,
        sep='\t',
        type_adjacency='binary',
        sparse_adjacency=True,
        symmetric_adjacency=True,
        train_batch_size=1024,
        test_batch_size=2048
):
    """
    Load train and test ratings for GNN-based models.
    Note that the user and item IDs are converted to sequential numbers.

    :param train_ratings_filepath: The training ratings CSV or TSV filepath.
    :param test_ratings_filepath: The test ratings CSV or TSV filepath.
    :param props_triples_filepath: The properties triples CSV or TSV filepath. It can be None, and it is used only if
                                   return_adjacency is True.
    :param sep: The separator to use for CSV or TSV files.
    :param type_adjacency: Used only if return_adjacency is True and sparse_adjacency is True. It can be either
                           'unary' for 1-only ratings, 'binary' for 0/1-only ratings and 'unary-kg' if you need both the
                           unary matrix and the KG unary graph. In the latter case it requires props and props_triples.
    :param sparse_adjacency: Whether to return the adjacency matrix as a sparse matrix instead of dense.
    :param symmetric_adjacency: Whether to return a symmetric adjacency matrix.
    :param train_batch_size: batch_size used in training phase.
    :param test_batch_size: batch_size used in test phase.
    :return: The training and test ratings data sequence for GNN-based models.
    """
    (train_ratings, test_ratings), (users, items), adj_matrix = \
        load_train_test_ratings(train_ratings_filepath,
                                test_ratings_filepath,
                                props_triples_filepath,
                                sep=sep,
                                return_adjacency=True,
                                type_adjacency=type_adjacency,
                                sparse_adjacency=sparse_adjacency,
                                symmetric_adjacency=symmetric_adjacency)
    data_train = UserItemGraphPosNegSample(
        train_ratings, users, items, adj_matrix,
        batch_size=train_batch_size
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
        props_triples_filepath=None,
        sep='\t',
        type_adjacency='unary',
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
    :param props_triples_filepath: The properties triples CSV or TSV filepath. It can be None, and it is used only if
                                   return_adjacency is True.
    :param bert_user_filepath: The filepath for User BERT embeddings.
    :param bert_item_filepath: The filepath for Item BERT embeddings.
    :param sep: The separator to use for CSV or TSV files.
    :param type_adjacency: Used only if return_adjacency is True and sparse_adjacency is True. It can be either
                           'unary' for 1-only ratings, 'binary' for 0/1-only ratings and 'unary-kg' if you need both the
                           unary matrix and the KG unary graph. In the latter case it requires props and props_triples.
    :param sparse_adjacency: Whether to return the adjacency matrix as a sparse matrix instead of dense.
    :param symmetric_adjacency: Whether to return a symmetric adjacency matrix.
    :param shuffle: Tells if shuffle the training dataset.
    :param train_batch_size: batch_size used in training phase.
    :param test_batch_size: batch_size used in test phase.
    :return: The training and test ratings data sequence for GNN-based models.
    """
    (train_ratings, test_ratings), (users, items), adj_matrix = \
        load_train_test_ratings(train_ratings_filepath,
                                test_ratings_filepath,
                                props_triples_filepath,
                                sep=sep,
                                return_adjacency=True,
                                type_adjacency=type_adjacency,
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
