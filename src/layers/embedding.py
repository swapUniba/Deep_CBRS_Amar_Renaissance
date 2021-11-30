import tensorflow as tf
from tensorflow import keras


class UserItemEmbedding(keras.layers.Layer):
    def __init__(self, users, items, **kwargs):
        super().__init__(**kwargs)
        self.users_embedding = {
            user_id: tf.Variable(tf.convert_to_tensor(embedding, dtype=tf.float32))
            for (user_id, embedding) in users.items()
        }
        self.items_embedding = {
            item_id: tf.Variable(tf.convert_to_tensor(embedding, dtype=tf.float32))
            for (item_id, embedding) in items.items()
        }

    def call(self, inputs):
        u = tf.stack([self.users_embedding[u.numpy()] for (u, _) in inputs])
        i = tf.stack([self.items_embedding[i.numpy()] for (_, i) in inputs])
        return u, i
