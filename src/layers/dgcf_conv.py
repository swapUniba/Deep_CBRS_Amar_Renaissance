from keras import layers

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.utils import gcn_filter

import tensorflow as tf


class DGCFConv(Conv):
    r"""
    A Deoscillated Adaptive Graph Collaborative Filtering (DGCF) from the homonym paper

    """

    def __init__(
        self,
        regularizer,
        **kwargs
    ):
        super().__init__(
        **kwargs
        )
        self.regularizer = regularizer
        self.locality_adaptive = LocalityAdaptive(regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def call(self, inputs, mask=None):
        x, a = inputs
        x = self.locality_adaptive(x)
        output = ops.modal_dot(a, x)
        return output

    @staticmethod
    def preprocess(a):
        # Crosshop matrix
        crosshop = ops.matrix_power(a, 2)

        # Laplacian matrix
        a, crosshop = gcn_filter(a), gcn_filter(crosshop)
        crosshop = DGCFConv.high_pass_filter(a, crosshop)

        # Final DGCF adjacency matrix
        return a + crosshop + tf.eye(a.shape[0])

    @staticmethod
    def high_pass_filter(adjacency, crosshop):
        """
        Apply annhigh pass filter to the crosshop matrix by filtering out values below a threshold epsilon.
        Epsilon is found by searching for the ratio closer to one.
        The ratio is between the base edges and the filtered crosshop

        :param adjacency: adjacency matrix
        :param crosshop: crosshop matrix
        :return: filtered crosshop matrix
        """
        epsilons = [1e-1, 1e-2, 1e-3, 5e-4]
        edges = tf.math.count_nonzero(adjacency)
        # Get filtered matrices for each epsilon
        cross_filtered = [tf.multiply(tf.where(crosshop > eps), crosshop) for eps in epsilons]
        # Count edges for filtered matrices
        cross_edges = [tf.math.count_nonzero(chp) for chp in cross_filtered]
        # Calculate the ratios
        ratios = [edges / chp_edges if edges > chp_edges else chp_edges / edges for chp_edges in cross_edges]
        best_ratio = tf.argmin(ratios)
        return cross_filtered[best_ratio]


class LocalityAdaptive(layers.Layer):
    """
    Computes Weighted sum of input tensor with learnable weights
    """
    def __init__(self, regularizer=None, **kwargs):
        super(LocalityAdaptive, self).__init__(**kwargs)
        self.w = None
        self.regularizer = regularizer

    def build(self, input_shape):
        n_nodes = len(input_shape)
        self.w = self.add_weight(
            name='locality-adaptive-weights',
            shape=[n_nodes, 1, 1],
            initializer='ones',
            regularizer=self.regularizer
        )

    def call(self, inputs, **kwargs):
        return tf.multiply(inputs, tf.sigmoid(self.w))
