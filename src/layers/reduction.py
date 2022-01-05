from tensorflow.keras import layers
import tensorflow as tf


class ReductionLayer(layers.Layer):
    """
    Reduces outputs from previous layers of the GNN
    """
    def __init__(self, method='concatenate', regularizer=None):
        """
        :param method: Defines the reduction method (concatenation, sum, mean, last, w-mean).
        :param regularizer: Fot w-sum only, regularizer of weights.
        """
        super(ReductionLayer, self).__init__()
        if method == 'concatenation':
            self.layer = layers.Concatenate()
        elif method == 'sum':
            self.layer = tf.add_n
        elif method == 'mean':
            self.layer = ReductionLayer.tensor_mean
        elif method == 'w-sum':
            self.layer = WeightedSum(regularizer)
        elif method == 'last':
            self.layer = lambda inputs: inputs[-1]
        else:
            raise ValueError('Reduction method not supported: ' + method)

    @classmethod
    def tensor_mean(cls, inputs):
        return tf.divide(tf.add_n(inputs), len(inputs))

    def call(self, inputs, **kwargs):
        return self.layer(inputs)


class WeightedSum(layers.Layer):
    """
    Computes Weighted sum of input tensor with learnable weights
    """
    def __init__(self, regularizer=None, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.w = None
        self.regularizer = regularizer

    def build(self, input_shape):
        n_layers = len(input_shape)
        self.w = self.add_weight(
            name='reduction-weights',
            shape=[n_layers, 1, 1],
            initializer='ones',
            regularizer=self.regularizer
        )

    def call(self, inputs, **kwargs):
        return tf.reduce_sum(tf.multiply(self.w * self.w, inputs), axis=0)
