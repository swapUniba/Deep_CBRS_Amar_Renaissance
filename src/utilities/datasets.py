import csv
import json
import numpy as np
import pandas as pd
import tensorflow as tf


class UserItemGenerator:
    def __init__(self, ratings_filepath, user_filepath, item_filepath):
        self.users, self.items, self.ratings = load_ratings(ratings_filepath)
        self.user_embeddings, self.item_embeddings = load_user_item_embeddings(user_filepath, item_filepath)

    def __len__(self):
        return len(self.users)

    def flow(self):
        for u, i, r in zip(self.users, self.items, self.ratings):
            yield (self.user_embeddings[u], self.item_embeddings[i]), (r,)

    def get_dataset(self):
        embedding_size = len(list(self.user_embeddings.values())[0])
        dataset = tf.data.Dataset.from_generator(
            self.flow, output_signature=(
                tf.TensorSpec(shape=(2, embedding_size), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.int64)
            )
        )
        return dataset


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
    return users, items, ratings
