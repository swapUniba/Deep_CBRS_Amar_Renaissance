import numpy as np
from tensorflow import keras

from models.basic import BasicCBRS
from utils.datasets import load_ratings
from utils.embeddings import load_user_item_embeddings


if __name__ == '__main__':
    user_embeddings, item_embeddings = load_user_item_embeddings(
        'embeddings/bert/user-lastlayer.json',
        'embeddings/bert/item-lastlayer.json'
    )
    users, items, ratings = load_ratings('datasets/movielens/train2id.tsv')

    model = BasicCBRS(user_embeddings, item_embeddings)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.build([(users[0], items[0])])
    model.summary()

    epochs = 10
    batch_size = 64
    model.fit(list(zip(users, items)), y=ratings, epochs=epochs, batch_size=batch_size, verbose=2)
