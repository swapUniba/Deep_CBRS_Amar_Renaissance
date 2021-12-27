import numpy as np


def top_k_metrics(ratings_true, ratings_pred, k=5):
    """
    Compute the macro-averaged Precision, Recall and F1 metrics @K.

    :param ratings_true: The actual ratings as an array of User-Item-Rating.
    :param ratings_pred: The predicted ratings as an array of User-Item-Rating.
    :param k: The K parameter.
    :return: Precision, Recall and F1 @K.
    """
    precisions, recalls = list(), list()
    user_indices = np.unique(ratings_true[:, 0])
    for user_idx in user_indices:
        user_mask = ratings_true[:, 0] == user_idx
        user_ratings_true = ratings_true[user_mask]
        user_ratings_pred = ratings_pred[user_mask]

        item_true_mask = user_ratings_true[:, 2] == 1
        items_true = user_ratings_true[item_true_mask, 1]

        relevant_idx = np.argsort(-user_ratings_pred[:, 2])
        relevant_items = user_ratings_pred[relevant_idx, 1]
        if k <= len(relevant_items):
            relevant_items = relevant_items[:k]

        prec = rec = 0.0
        items_true = set(items_true)
        relevant_items = set(relevant_items)
        if relevant_items:
            prec = len(items_true & relevant_items) / len(relevant_items)
        if items_true:
            rec = len(items_true & relevant_items) / len(items_true)

        precisions.append(prec)
        recalls.append(rec)

    precision, recall = np.mean(precisions), np.mean(recalls)
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def top_predictions(predictions, user_ids, item_ids, k=5):
    """
    Compute the Top-K suggested items for each user, given prediction scores.

    :param predictions: The prediction scores as an array that associate to each user and item a score between 0 and 1.
    :param user_ids: An array of user ids.
    :param item_ids: An array of item ids.
    :param k: The K parameter.
    :return: An array that associate to each user a sequence of K suggested items.
    """
    predictions = predictions.reshape(len(user_ids), len(item_ids))
    sort_idx = np.argsort(-predictions, axis=1)
    item_ids = np.tile(item_ids, reps=(len(user_ids), 1))
    user_idx = np.expand_dims(np.arange(len(user_ids)), axis=1)
    return item_ids[user_idx, sort_idx][:, :k]
