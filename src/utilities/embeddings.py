import numpy as np
import pandas as pd


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
