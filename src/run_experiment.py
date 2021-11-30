import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.basic import BasicCBRS
from utilities.datasets import UserItemGenerator


if __name__ == '__main__':
    dataset = UserItemGenerator(
        'datasets/movielens/train2id.tsv', 'embeddings/bert/user-lastlayer.json', 'embeddings/bert/item-lastlayer.json'
    ).get_dataset().batch(batch_size=64)

    model = BasicCBRS()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(dataset, epochs=1, verbose=2)
