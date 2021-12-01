import csv
import json
import numpy as np
import pandas as pd
import tensorflow as tf
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
        return len(self.ratings) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = idx * self.batch_size
        indexes = self.indexes[batch_idx:batch_idx + self.batch_size]
        ratings = self.ratings[indexes]
        user_embeddings = np.stack([self.user_embeddings[u] for u in ratings[:, 0]])
        item_embeddings = np.stack([self.item_embeddings[i] for i in ratings[:, 1]])
        return (user_embeddings, item_embeddings), ratings[:, 2]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ratings))
        if self.shuffle:
            self.random_state.shuffle(self.indexes)


class BERTUserItemSequence(BaseUserItemSequence):
    def __init__(self, ratings_filepath, user_filepath, item_filepath, batch_size=512, shuffle=True, seed=42):
        ratings = load_ratings(ratings_filepath)
        user_embeddings, item_embeddings = load_bert_user_item_embeddings(user_filepath, item_filepath)
        super().__init__(
            ratings, user_embeddings, item_embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )


class GraphUserItemSequence(BaseUserItemSequence):
    def __init__(self, ratings_filepath, filepath, batch_size=512, shuffle=True, seed=42):
        ratings = load_ratings(ratings_filepath)
        embeddings = load_graph_user_item_embeddings(filepath)
        super().__init__(
            ratings, embeddings, embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )


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
