import tensorflow as tf

from spektral.layers.convolutional.conv import Conv


class KGATConv(Conv):
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
        self.return_attn_coef = return_attn_coef
        self.output_dim = self.channels

    def build(self, input_shape):
        assert len(input_shape) == 2
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
        x, attn_coef = inputs

        # Compute a weighted sum using the attention values
        attn_neigh = tf.sparse.sparse_dense_matmul(attn_coef, x)

        # Compute the final nodes embeddings (GCN way)
        output = tf.matmul(attn_neigh, self.kernel)

        # Apply bias and activation function
        if self.use_bias:
            output += self.bias
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        # Apply L2 normalization
        output = tf.math.l2_normalize(output, axis=1)

        if self.return_attn_coef:
            return output, attn_coef
        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
        }
