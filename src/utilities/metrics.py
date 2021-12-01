import numpy as np


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
