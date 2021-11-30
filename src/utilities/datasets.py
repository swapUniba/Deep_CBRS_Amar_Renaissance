import csv
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class UserItemSequence(keras.utils.Sequence):
    def __init__(self, ratings_filepath, user_filepath, item_filepath, batch_size=512):
        self.users, self.items, self.ratings = load_ratings(ratings_filepath)
        self.user_embeddings, self.item_embeddings = load_user_item_embeddings(user_filepath, item_filepath)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.users) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = idx * self.batch_size
        users = self.users[batch_idx:batch_idx + self.batch_size]
        items = self.items[batch_idx:batch_idx + self.batch_size]
        ratings = self.ratings[batch_idx:batch_idx + self.batch_size]
        user_embeddings = np.stack([self.user_embeddings[u] for u in users])
        item_embeddings = np.stack([self.item_embeddings[i] for i in items])
        return (user_embeddings, item_embeddings), ratings


def load_bert_embeddings(filepath):
    embeddings = pd.read_json(filepath)
    return embeddings.sort_values(by=['ID_OpenKE'])


def load_graph_embeddings(filepath):
    with open(filepath) as fp:
        embeddings = json.load(fp)
    return embeddings['ent_embeddings']


def load_user_item_embeddings(user_filepath, item_filepath, source='bert'):
    if source == 'bert':
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

    raise ValueError("Unknown embbedding source called: {}".format(source))


def load_ratings(filepath):
    users, items, ratings = [], [], []
    with open(filepath) as fp:
        csv_reader = csv.reader(fp, delimiter='\t')
        for row in csv_reader:
            users.append(int(row[0]))
            items.append(int(row[1]))
            ratings.append(int(row[2]))
    users = np.array(users, dtype=np.int32)
    items = np.array(items, dtype=np.int32)
    ratings = np.array(ratings, dtype=np.int32)
    return users, items, ratings
