import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.basic import BasicCBRS
from utilities.datasets import BERTUserItemSequence, GraphUserItemSequence


if __name__ == '__main__':
    """
    dataset = BERTUserItemSequence(
        'datasets/movielens/train2id.tsv',
        'embeddings/bert/user-lastlayer.json',
        'embeddings/bert/item-lastlayer.json',
        batch_size=512, shuffle=True
    )
    """

    dataset = GraphUserItemSequence(
        'datasets/movielens/train2id.tsv',
        'embeddings/user-item-properties/768DistMult.json',
        batch_size=512, shuffle=True
    )

    model = BasicCBRS()
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(dataset, epochs=25, workers=4)
