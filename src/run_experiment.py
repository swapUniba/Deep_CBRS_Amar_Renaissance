import numpy as np
from tensorflow import keras

from models.hybrid import HybridCBRS
from utilities.datasets import HybridUserItemEmbeddings, load_train_test_ratings
from utilities.datasets import load_bert_user_item_embeddings, load_graph_user_item_embeddings
from utilities.metrics import top_k_metrics


if __name__ == '__main__':
    # Load train and test ratings
    (train_ratings, test_ratings), (users, items) = load_train_test_ratings(
        'datasets/movielens/train2id.tsv',
        'datasets/movielens/test2id.tsv'
    )

    
    # Load both Graph and BERT user/item embeddings
    graph_user_embeddings, graph_item_embeddings = load_graph_user_item_embeddings(
        'embeddings/user-item-properties/768DistMult.json',
        users, items
    )
    bert_user_embeddings, bert_item_embeddings = load_bert_user_item_embeddings(
        'embeddings/bert/user-lastlayer.json',
        'embeddings/bert/item-lastlayer.json',
        users, items
    )

    # Setup both training and test data sequences
    data_train = HybridUserItemEmbeddings(
        train_ratings, graph_user_embeddings, graph_item_embeddings, bert_user_embeddings, bert_item_embeddings,
        batch_size=512, shuffle=True
    )
    data_test = HybridUserItemEmbeddings(
        test_ratings, graph_user_embeddings, graph_item_embeddings, bert_user_embeddings, bert_item_embeddings,
        batch_size=2048, shuffle=False
    )

    # Instantiate, train and evaluate a hybrid feature-based recommender system
    model = HybridCBRS(feature_based=True)
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
