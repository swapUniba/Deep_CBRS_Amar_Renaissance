import tensorflow as tf

from spektral.layers.convolutional import GCNConv
from spektral.layers.convolutional.conv import Conv


class KGCNConv(Conv):
    def __init__(
        self,
        n_users,
        n_items,
        channels,
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
        self.n_users = n_users
        self.n_items = n_items
        self.channels = channels
        self.output_dim = channels

        # Initialize two GCNs (Bi + KG)
        gcn_kwargs = {
            'activation': activation,
            'use_bias': use_bias,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'activity_regularizer': activity_regularizer,
            'kernel_constraint': kernel_constraint,
            'bias_constraint': bias_constraint
        }
        self.gcn_bi = GCNConv(channels, **gcn_kwargs)
        self.gcn_kg = GCNConv(channels, **gcn_kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 5
        assert input_shape[0][1] == input_shape[1][1]
        assert input_shape[0][1] == input_shape[2][1]
        assert input_shape[3][0] == input_shape[0][0] + input_shape[1][0]
        assert input_shape[4][0] == input_shape[1][0] + input_shape[2][0]
        input_dim = input_shape[0][1]

        bi_input_shape = (input_shape[0][0] + input_shape[1][0], input_shape[0][1]), input_shape[3]
        kg_input_shape = (input_shape[1][0] + input_shape[2][0], input_shape[0][1]), input_shape[4]
        self.gcn_bi.build(bi_input_shape)
        self.gcn_kg.build(kg_input_shape)

        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_dim * 2, self.output_dim],
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

    def call(self, inputs, **kwargs):
        u_e, i_e, p_e, a_bi, a_kg = inputs

        # Pass through Bi and KG GCN models
        x_bi = tf.concat([u_e, i_e], axis=0)
        x_kg = tf.concat([i_e, p_e], axis=0)
        x_bi = self.gcn_bi([x_bi, a_bi])
        x_kg = self.gcn_kg([x_kg, a_kg])

        # Re-obtain the user and properties embeddings
        # The item embeddings are obtained by a non-linear combination of both Bi and KG GCN models
        u_e = x_bi[:self.n_users]
        p_e = x_kg[self.n_items:]
        i_e = tf.concat([x_bi[self.n_users:], x_kg[:self.n_items]], axis=1)
        i_e = tf.matmul(i_e, self.kernel)
        if self.use_bias:
            i_e += self.bias
        i_e = self.activation(i_e)

        return u_e, i_e, p_e

    @property
    def config(self):
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "channels": self.channels
        }
