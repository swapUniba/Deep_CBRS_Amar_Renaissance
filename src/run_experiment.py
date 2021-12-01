from tensorflow import keras
from models.basic import BasicCBRS
from models.hybrid import HybridCBRS
from utilities.datasets import HybridUserItemSequence, GraphUserItemSequence, BERTUserItemSequence


if __name__ == '__main__':
    dataset = HybridUserItemSequence(
        'datasets/movielens/train2id.tsv',
        'embeddings/user-item-properties/768DistMult.json',
        'embeddings/bert/user-lastlayer.json',
        'embeddings/bert/item-lastlayer.json',
        batch_size=512, shuffle=True
    )

    model = HybridCBRS(feature_based=False)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(dataset, epochs=25, workers=4)
