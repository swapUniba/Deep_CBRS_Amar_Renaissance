import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.basic import BasicCBRS
from utilities.datasets import UserItemGenerator


if __name__ == '__main__':
    generator = UserItemGenerator(
        'datasets/movielens/train2id.tsv', 'embeddings/bert/user-lastlayer.json', 'embeddings/bert/item-lastlayer.json'
    )
    dataset = generator.get_dataset()

    model = BasicCBRS()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    epochs = 25
    batch_size = 1024
    dataset = dataset.repeat().batch(batch_size, num_parallel_calls=4)
    steps_per_epoch = len(generator) // batch_size
    model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs)
