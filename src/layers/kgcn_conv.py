import tensorflow as tf
from spektral.layers import ops

from spektral.layers.convolutional.conv import Conv


class KGCNConv(Conv):
    def __init__(
        self,
        channels,
        return_attn_coef=False,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.channels = channels
        self.output_dim = channels
        self.return_attn_coef = return_attn_coef

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[0][1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_dim, self.output_dim],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[self.output_dim],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name="bias",
            )
        self.built = True

    def call(self, inputs, mask=None):
        x, r, a = inputs

        # Get the sources, targets and relation values from the adjacency sparse matrix
        indices, values = a.indices, a.values
        targets, sources = indices[:, 1], indices[:, 0]

        # Compute the scores between entities and relations using a dot product (Eq. 1)
        pi = tf.einsum('hi,ki->hk', x, r)
        attn_coef = tf.gather_nd(pi, tf.stack([sources, values], axis=1))

        # Compute the attention coefficients by softmax w.r.t. the source entities (Eq. 3)
        attn_coef = ops.unsorted_segment_softmax(attn_coef, sources, x.shape[0])
        attn_coef = tf.expand_dims(attn_coef, axis=1)

        # Compute the weighted sum of neighbours using attention coefficients (Eq. 2)
        output = attn_coef * tf.gather(x, targets)
        output = tf.math.unsorted_segment_sum(output, sources, x.shape[0])

        # Compute the final nodes embeddings (GCN way)
        output = tf.matmul(output + x, self.kernel)

        # Apply bias and activation function
        if self.use_bias:
            output += self.bias
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        if self.return_attn_coef:
            return output, attn_coef
        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "return_attn_coef": self.return_attn_coef
        }
