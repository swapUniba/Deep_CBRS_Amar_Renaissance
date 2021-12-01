import csv
import json
import numpy as np
import pandas as pd
from tensorflow import keras


class BaseUserItemSequence(keras.utils.Sequence):
    def __init__(
        self,
        ratings,
        user_embeddings,
        item_embeddings,
        batch_size=512,
        shuffle=True,
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
        self.ratings = ratings
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(seed)
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """
        Get the number of full batches.

        :return: The number of full batches.
        """
        return len(self.ratings) // self.batch_size

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item embeddings and the rating.

        :param idx: THe index of the batch.
        :return: A pair consisting of User-Item embeddings and the ratings.
        """
        batch_idx = idx * self.batch_size
        indexes = self.indexes[batch_idx:batch_idx + self.batch_size]
        ratings = self.ratings[indexes]
        user_embeddings = np.stack([self.user_embeddings[u] for u in ratings[:, 0]])
        item_embeddings = np.stack([self.item_embeddings[i] for i in ratings[:, 1]])
        return (user_embeddings, item_embeddings), ratings[:, 2]

    def on_epoch_end(self):
        """
        Shuffles the indexes at the end of every epoch.
        """
        self.indexes = np.arange(len(self.ratings))
        if self.shuffle:
            self.random_state.shuffle(self.indexes)


class BERTUserItemSequence(BaseUserItemSequence):
    def __init__(self, ratings_filepath, user_filepath, item_filepath, batch_size=512, shuffle=True, seed=42):
        """
        Initialize a User-Item embeddings sequence with BERT embeddings.

        :param ratings_filepath: The filepath of the ratings file.
        :param user_filepath: The filepath of user BERT embeddings.
        :param item_filepath: The filepath of item BERT embeddings.
        :param batch_size: The batch size.
        :param shuffle: Whether to shuffle the sequence.
        :param seed: The seed value used to shuffle the sequence.
        """
        ratings = load_ratings(ratings_filepath)
        user_embeddings, item_embeddings = load_bert_user_item_embeddings(user_filepath, item_filepath)
        super().__init__(
            ratings, user_embeddings, item_embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )


class GraphUserItemSequence(BaseUserItemSequence):
    def __init__(self, ratings_filepath, graph_filepath, batch_size=512, shuffle=True, seed=42):
        """
        Initialize a User-Item embeddings sequence with Graph embeddings.

        :param ratings_filepath: The filepath of the ratings file.
        :param graph_filepath: The filepath of user and item Graph embeddings.
        :param batch_size: The batch size.
        :param shuffle: Whether to shuffle the sequence.
        :param seed: The seed value used to shuffle the sequence.
        """
        ratings = load_ratings(ratings_filepath)
        embeddings = load_graph_user_item_embeddings(graph_filepath)
        super().__init__(
            ratings, embeddings, embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )


class HybridUserItemSequence(keras.utils.Sequence):
    def __init__(
        self,
        ratings_filepath,
        graph_filepath,
        user_filepath,
        item_filepath,
        batch_size=512,
        shuffle=True,
        seed=42
    ):
        """
        Initialize a User-Item embeddings sequence with both Graph and BERT embeddings.

        :param ratings_filepath: The filepath of the ratings file.
        :param graph_filepath: The filepath of user and item Graph embeddings.
        :param user_filepath: The filepath of user BERT embeddings.
        :param item_filepath: The filepath of item BERT embeddings.
        :param batch_size: The batch size.
        :param shuffle: Whether to shuffle the sequence.
        :param seed: The seed value used to shuffle the sequence.
        """
        ratings = load_ratings(ratings_filepath)
        graph_embeddings = load_graph_user_item_embeddings(graph_filepath)
        user_embeddings, item_embeddings = load_bert_user_item_embeddings(user_filepath, item_filepath)
        self.graph_sequence = BaseUserItemSequence(
            ratings, graph_embeddings, graph_embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )
        self.bert_sequence = BaseUserItemSequence(
            ratings, user_embeddings, item_embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )

    def __len__(self):
        """
        Get the number of full batches.

        :return: The number of full batches.
        """
        return len(self.graph_sequence)

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item Graph and BERT embeddings, and the rating.

        :param idx: THe index of the batch.
        :return: A tuple consisting of User-Item Graph and BERT embeddings and the ratings.
        """
        (user_graph_embeddings, item_graph_embeddings), ratings = self.graph_sequence[idx]
        (user_bert_embeddings, item_bert_embeddings), ratings1 = self.bert_sequence[idx]
        return (user_graph_embeddings, item_graph_embeddings, user_bert_embeddings, item_bert_embeddings), ratings

    def on_epoch_end(self):
        """
        Shuffles the indexes at the end of every epoch.
        """
        self.graph_sequence.on_epoch_end()
        self.bert_sequence.on_epoch_end()


def load_bert_embeddings(filepath):
    embeddings = pd.read_json(filepath)
    return embeddings.sort_values(by=['ID_OpenKE'])


def load_graph_embeddings(filepath):
    with open(filepath) as fp:
        embeddings = json.load(fp)
    return embeddings['ent_embeddings']


def load_bert_user_item_embeddings(user_filepath, item_filepath):
    user_embeddings, item_embeddings = dict(), dict()
    df_users = load_bert_embeddings(user_filepath)
    df_items = load_bert_embeddings(item_filepath)
    for _, user in df_users.iterrows():
        user_id = user['ID_OpenKE']
        user_embeddings[user_id] = np.array(user['profile_embedding'], dtype=np.float32)
    for _, item in df_items.iterrows():
        item_id = item['ID_OpenKE']
        item_embeddings[item_id] = np.array(item['embedding'], dtype=np.float32)
    return user_embeddings, item_embeddings


def load_graph_user_item_embeddings(filepath):
    return np.array(load_graph_embeddings(filepath), dtype=np.float32)


def load_ratings(filepath):
    ratings = []
    with open(filepath) as fp:
        csv_reader = csv.reader(fp, delimiter='\t')
        for row in csv_reader:
            ratings.append((int(row[0]), int(row[1]), int(row[2])))
    ratings = np.array(ratings, dtype=np.int32)
    return ratings
