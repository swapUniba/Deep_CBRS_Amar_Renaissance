import tensorflow as tf
from tensorflow.keras import backend as K

from spektral.layers.convolutional.conv import Conv


class KGATConv(Conv):
    def __init__(
        self,
        channels,
        return_attn_coef=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super().__init__(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            **kwargs
        )
        self.channels = channels
        self.return_attn_coef = return_attn_coef

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1][-1]
        num_kernels = input_shape[0][0]

        self.kernel = self.add_weight(
            name="kernel",
            shape=[num_kernels, input_dim, self.channels],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.built = True

    def call(self, inputs):
        r, x, a = inputs

        # Collect all the attention adjacency matrices, for all the relation types
        attn_coef = []
        for i in range(len(a)):
            attn_coef.append(self._call_single(r, x, a[i], self.kernel[i]))

        # Sum all the attention adjacency matrices and apply the softmax on the neighbours-related attentions
        attn_coef = K.sum(attn_coef, axis=0)
        attn_coef = K.softmax(attn_coef, axis=1)

        # Compute a weighted sum using the attention values
        output = K.dot(attn_coef, x)

        if self.return_attn_coef:
            return output, attn_coef
        return output

    def _call_single(self, r, x, a, kernel):
        indices = a.indices
        targets, sources = indices[:, 1], indices[:, 0]

        # Apply the weight matrix
        y = K.dot(x, kernel)

        # Compute the attention values for all the observed interactions in the adjacency matrix
        h = tf.gather(y, targets)
        t = tf.gather(tf.tanh(y + tf.expand_dims(r, axis=0)), sources)
        pi = K.batch_dot(h, t)

        # Rebuild the adjacency matrix, but with attention as values
        return tf.sparse.SparseTensor(
            indices=a.indices,
            values=pi,
            dense_shape=a.shape
        )

    @property
    def config(self):
        return {
            "channels": self.channels,
        }
