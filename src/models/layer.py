from tensorflow.keras import layers
import tensorflow as tf


class ReductionLayer(layers.Layer):
    """
    Reduces outputs from previous layers of the GNN
    """
    def __init__(self, method='concatenate'):
        """
        :param method: Defines the reduction method (concatenation, sum, mean, last, w-mean).
        """
        super(ReductionLayer, self).__init__()
        if method == 'concatenation':
            self.layer = layers.Concatenate()
        elif method == 'sum':
            self.layer = tf.add_n
        elif method == 'mean':
            self.layer = ReductionLayer.tensor_mean
        elif method == 'w-mean':
            raise NotImplementedError('Weighted mean have to be implemented')
        elif method == 'last':
            self.layer = lambda inputs: inputs[-1]
        else:
            raise ValueError('Reduction method not supported: ' + method)

    @classmethod
    def tensor_mean(cls, inputs):
        return tf.divide(tf.add_n(inputs), len(inputs))

    def call(self, inputs):
        return self.layer(inputs)
