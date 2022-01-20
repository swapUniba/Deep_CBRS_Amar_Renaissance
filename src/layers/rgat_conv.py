import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import Dropout

from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.layers.ops import modes


class GATConv(Conv):
    def __init__(
        self,
        channels,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        attn_kernel_constraint=None,
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
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.output_dim = self.channels

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
        n_nodes = self.kernel.shape[1]
        r, x, a = inputs
        attn_info = []
        for i in range(len(a)):
            attn_info.append(self._call_single(r, x, a[i], self.kernel[i]))

        # WHAT DO ???

        output = tf.reduce_sum(output, axis=-2)

        if self.use_bias:
            output += self.bias
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def _call_single(self, r, x, a, kernel):
        # Prepare message-passing
        indices = a.indices
        targets, sources = indices[:, 1], indices[:, 0]

        y = K.dot(x, kernel)

        h = tf.gather(y, targets)  # [n_interations, n_channels]
        t = tf.gather(tf.tanh(y + tf.expand_dims(r, axis=0)), sources)
        pi = tf.tensordot(h, t)

        return pi, targets, sources

    @property
    def config(self):
        return {
            "channels": self.channels,
        }
