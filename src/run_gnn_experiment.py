import numpy as np
from tensorflow import keras

from models.gnn import BasicGCN
from utilities.datasets import UserItemGraph, load_train_test_ratings
from utilities.metrics import top_k_metrics


if __name__ == '__main__':
    # Load train and test ratings
    (train_ratings, test_ratings), adj_matrix = load_train_test_ratings(
        'datasets/movielens/train2id.tsv',
        'datasets/movielens/test2id.tsv',
        return_adjacency=True, binary_adjacency=False, sparse_adjacency=True
    )

    # Setup both training and test data sequences
    data_train = UserItemGraph(
        train_ratings, adj_matrix,
        batch_size=1024, shuffle=True
    )
    data_test = UserItemGraph(
        test_ratings, adj_matrix,
        batch_size=2048, shuffle=False
    )

    # Instantiate, train and evaluate a hybrid feature-based recommender system
    model = BasicGCN(data_train.adj_matrix)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(data_train, epochs=25)
    model.evaluate(data_test)

    # Compute Precision, Recall and F1 @K metrics
    predictions = model.predict(data_test)
    ratings_pred = np.concatenate([test_ratings[:, [0, 1]], predictions], axis=1)
    print('P@ 5, R@ 5, F@ 5: {}'.format(top_k_metrics(test_ratings, ratings_pred, k= 5)))
    print('P@10, R@10, F@10: {}'.format(top_k_metrics(test_ratings, ratings_pred, k=10)))
    print('P@20, R@20, F@20: {}'.format(top_k_metrics(test_ratings, ratings_pred, k=20)))
