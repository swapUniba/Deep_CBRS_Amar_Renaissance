from tensorflow import keras
import tensorflow as tf


class BPRLoss(keras.losses.Loss):
    def __init__(self, name="BPR_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        half = len(y_pred) // 2
        return - tf.reduce_sum(
            tf.math.log(
                tf.math.sigmoid(
                    y_pred[0:half] - y_pred[half:]
                )
            )
        )