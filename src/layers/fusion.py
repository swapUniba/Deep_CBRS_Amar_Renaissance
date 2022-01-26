from tensorflow.keras import layers
import tensorflow as tf


class FusionLayer(layers.Layer):
    def __init__(self, method='concatenate'):
        """
        Custom Fusion layer.

        :param method: The fusion method. It can be either 'concatenate' for straightforward concatenation along the
                       features' axis, or 'attention' for attention-based summation of features.
        """
        super().__init__()
        if method not in ['concatenate', 'attention']:
            raise ValueError("Unknown concatenation method called {}".format(method))
        self.method = method
        self.proj_first = None

    def build(self, input_shape):
        a_shape, b_shape = input_shape
        assert len(a_shape) == 2 and len(b_shape) == 2
        if self.method == 'attention':
            if a_shape[1] != b_shape[1]:
                if a_shape[1] > b_shape[1]:
                    att_weight_shape = (a_shape[1], a_shape[1])
                    proj_weight_shape = (b_shape[1], a_shape[1])
                    self.proj_first = False
                else:
                    att_weight_shape = (b_shape[1], b_shape[1])
                    proj_weight_shape = (a_shape[1], b_shape[1])
                    self.proj_first = True

                # The projection weight is used only if the features dimensions are incompatible
                # That is, the smaller features vector is projected into a higher dimension
                self.proj_weight = self.add_weight(
                    name='proj_weight',
                    shape=proj_weight_shape,
                    initializer='glorot_uniform'
                )
            else:
                att_weight_shape = (a_shape[1], a_shape[1])
                self.proj_first = None
            self.att_weight = self.add_weight(
                name='att_weight',
                shape=att_weight_shape,
                initializer='glorot_uniform'
            )

    def call(self, inputs, *args, **kwargs):
        a, b = inputs
        if self.method == 'concatenate':
            # Do basic concatenation along the features' axis
            return tf.concat([a, b], axis=1)
        if self.method == 'attention':
            # Project the smaller features vector
            if self.proj_first is not None:
                if self.proj_first:
                    a = tf.matmul(a, self.proj_weight)
                else:
                    b = tf.matmul(b, self.proj_weight)

            # Compute the attention coefficients
            x = tf.stack([a, b], axis=1)
            att_coef = tf.tanh(tf.matmul(x, self.att_weight))
            att_coef = tf.math.softmax(att_coef, axis=1)

            # Apply a weighted summation of the features
            return tf.reduce_sum(att_coef * x, axis=1)
