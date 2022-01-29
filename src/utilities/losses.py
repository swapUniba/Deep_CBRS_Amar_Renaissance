from tensorflow import keras
import tensorflow as tf


class BPRLoss(keras.losses.Loss):
    """
    The Bayesian Personalized Ranking loss (BPRLoss).

    It assumes that the scores of observed data are in the first half of the batch,
    while the scores of un-observed data (e.g. by negative sampling) are in the last half of the batch.
    """
    def __init__(self, name="BPR_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        if len(y_pred) % 2 != 0:
            y_pred = y_pred[:-1]
        half = len(y_pred) // 2
        return - tf.reduce_mean(
            tf.math.log(
                tf.math.sigmoid(
                    y_pred[:half] - y_pred[half:]
                )
            )
        )
