import csv
import json
import numpy as np
import pandas as pd
from tensorflow import keras


class UserItemEmbeddings(keras.utils.Sequence):
    def __init__(
        self,
        ratings,
        user_embeddings,
        item_embeddings,
        batch_size=512,
        shuffle=False,
        seed=42
    ):
        """
        A base class for User-Item embeddings sequence.

        :param ratings: A numpy array of triples (UserID, ItemID, Rating).
        :param user_embeddings: A dictionary of UserID and associated embedding.
        :param item_embeddings: A dictionary of ItemID and associated embedding.
        :param batch_size: The batch size.
        :param shuffle: Whether to shuffle the sequence.
        :param seed: The seed value used to shuffle the sequence.
        """
        super().__init__()

        # Set the ratings
        self.ratings = ratings

        # Set the embeddings
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

        # Set other settings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indexes = None
        self.random_state = None
        self.on_epoch_end()

    def get_users(self):
        return np.unique(self.ratings[:, 0])

    def get_items(self):
        return np.unique(self.ratings[:, 1])

    def __len__(self):
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return int(np.ceil(len(self.ratings) / self.batch_size))

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item embeddings and the rating.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item embeddings and the ratings.
        """
        batch_idx = idx * self.batch_size
        batch_off = min(batch_idx + self.batch_size, len(self.ratings))
        if self.shuffle:
            ratings = self.ratings[self.indexes[batch_idx:batch_off]]
        else:
            ratings = self.ratings[batch_idx:batch_off]
        user_embeddings = self.user_embeddings[ratings[:, 0]]
        item_embeddings = self.item_embeddings[ratings[:, 1]]
        return (user_embeddings, item_embeddings), ratings[:, 2]

    def on_epoch_end(self):
        """
        Shuffles the indexes at the end of every epoch.
        """
        if self.shuffle:
            if self.random_state is None:
                self.random_state = np.random.RandomState(self.seed)
            self.indexes = np.arange(len(self.ratings))
            self.random_state.shuffle(self.indexes)


class HybridUserItemEmbeddings(keras.utils.Sequence):
    def __init__(
        self,
        ratings,
        graph_user_embeddings,
        graph_item_embeddings,
        bert_user_embeddings,
        bert_item_embeddings,
        batch_size=512,
        shuffle=False,
        seed=42
    ):
        self.graph_embeddings = UserItemEmbeddings(
            ratings, graph_user_embeddings, graph_item_embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )
        self.bert_embeddings = UserItemEmbeddings(
            ratings, bert_user_embeddings, bert_item_embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )

    def __len__(self):
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return len(self.graph_embeddings)

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item embeddings and the rating.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item embeddings and the ratings.
        """
        (user_graph_embeddings, item_graph_embeddings), ratings = self.graph_embeddings[idx]
        (user_bert_embeddings, item_bert_embeddings), _ = self.bert_embeddings[idx]
        return (user_graph_embeddings, item_graph_embeddings, user_bert_embeddings, item_bert_embeddings), ratings

    def on_epoch_end(self):
        """
        Calls on_epoch_end() to any sub-sequence.
        """
        self.graph_embeddings.on_epoch_end()
        self.bert_embeddings.on_epoch_end()


class UserItemGraph(keras.utils.Sequence):
    def __init__(
        self,
        ratings,
        adj_matrix,
        batch_size=512,
        shuffle=False,
        seed=42
    ):
        super().__init__()

        # Set the ratings and the adjency matrix
        self.ratings = ratings
        self.adj_matrix = adj_matrix

        # Set other settings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indexes = None
        self.random_state = None
        self.on_epoch_end()

    def get_users(self):
        return np.unique(self.ratings[:, 0])

    def get_items(self):
        return np.unique(self.ratings[:, 1])

    def __len__(self):
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return int(np.ceil(len(self.ratings) / self.batch_size))

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item embeddings and the rating.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item embeddings and the ratings.
        """
        batch_idx = idx * self.batch_size
        batch_off = min(batch_idx + self.batch_size, len(self.ratings))
        if self.shuffle:
            ratings = self.ratings[self.indexes[batch_idx:batch_off]]
        else:
            ratings = self.ratings[batch_idx:batch_off]
        return (ratings[:, 0], ratings[:, 1], self.adj_matrix), ratings[:, 2]

    def on_epoch_end(self):
        """
        Shuffles the indexes at the end of every epoch.
        """
        if self.shuffle:
            if self.random_state is None:
                self.random_state = np.random.RandomState(self.seed)
            self.indexes = np.arange(len(self.ratings))
            self.random_state.shuffle(self.indexes)


def load_train_test_ratings(train_filepath, test_filepath, return_adjacency=False):
    """
    Load train and test ratings. Note that the user and item IDs are converted to sequential numbers.

    :param train_filepath: The training ratings CSV filepath.
    :param test_filepath: The test ratings CSV filepath.
    :return: The training and test ratings as an array of User-Item-Rating where IDs are made sequential.
             Moreover, it returns the users and items original unique IDs if return_adjacency is False,
             otherwise it returns the training interactions adjacency matrix (assuming un-directed arcs).
    """
    # Load the ratings arrays
    train_ratings = pd.read_csv(train_filepath).to_numpy()
    test_ratings = pd.read_csv(test_filepath).to_numpy()

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

    # Compute the adjacency matrix (actually two, for both positive and negative ratings)
    adj_size = len(users) + len(items)
    pos_idx = train_ratings[:, 2] == 1
    neg_idx = ~pos_idx
    adj_matrix = np.zeros([2, adj_size, adj_size], dtype=np.int32)
    adj_matrix[0, train_ratings[pos_idx, 0], train_ratings[pos_idx, 1]] = 1
    adj_matrix[1, train_ratings[neg_idx, 0], train_ratings[neg_idx, 1]] = 1
    adj_matrix += np.transpose(adj_matrix, axes=[0, 2, 1])

    return (train_ratings, test_ratings), adj_matrix


def load_graph_embeddings(filepath):
    with open(filepath) as fp:
        embeddings = json.load(fp)
    return embeddings['ent_embeddings']


def load_bert_embeddings(filepath):
    embeddings = pd.read_json(filepath)
    return embeddings.sort_values(by=['ID_OpenKE'])


def load_graph_user_item_embeddings(filepath, users, items):
    graph_embeddings = np.array(load_graph_embeddings(filepath), dtype=np.float32)
    user_embeddings = graph_embeddings[users]
    item_embeddings = graph_embeddings[items]
    return user_embeddings, item_embeddings


def load_bert_user_item_embeddings(user_filepath, item_filepath, users, items):
    user_embeddings, item_embeddings = dict(), dict()
    df_users = load_bert_embeddings(user_filepath)
    df_items = load_bert_embeddings(item_filepath)
    for _, user in df_users.iterrows():
        user_id = user['ID_OpenKE']
        user_embeddings[user_id] = np.array(user['profile_embedding'], dtype=np.float32)
    for _, item in df_items.iterrows():
        item_id = item['ID_OpenKE']
        item_embeddings[item_id] = np.array(item['embedding'], dtype=np.float32)
    user_embeddings = np.stack([user_embeddings[u] for u in users])
    item_embeddings = np.stack([item_embeddings[i] for i in items])
    return user_embeddings, item_embeddings
