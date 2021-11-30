import tensorflow as tf
from tensorflow import keras

from layers.embedding import UserItemEmbedding


class BasicCBRS(keras.Model):
    def __init__(self, users, items):
        super().__init__()
        self.embedding = UserItemEmbedding(users, items)
        self.unet = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense( 64, activation='relu')
        ])
        self.inet = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense( 64, activation='relu')
        ])
        self.concat = keras.layers.Concatenate()
        self.fc = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])


    def call(self, inputs):
        print(inputs)
        u, i = self.embedding(inputs)
        u = self.unet(u)
        i = self.inet(i)
        x = self.concat([u, i])
        return self.fc(x)
