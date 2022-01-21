import tensorflow as tf

from layers.kconv import KConv


class KGATConv(KConv):
    def __init__(
        self,
        channels,
        n_folds=100,
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
        self.n_folds = n_folds
        self.return_attn_coef = return_attn_coef
        self.output_dim = self.channels
        self.adj_shape = None

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[0][1]
        num_kernels = input_shape[1][0]

        self.adj_shape = input_shape[2]
        self.rel_kernel = self.add_weight(
            name="rel_kernel",
            shape=[num_kernels, input_dim, self.output_dim],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=[self.output_dim, self.output_dim],
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

        # The adjacency matrix is assumed to be only one such that
        # the value of a_ij is the relation id of an edge from entity i to entity j
        indices, values = a.indices, a.values
        targets, sources = indices[:, 1], indices[:, 0]
        unsq_x = tf.expand_dims(x, axis=1)

        # Computes the attention values for all the observed interactions
        pis = []
        len_fold = len(values) // self.n_folds
        for i in range(self.n_folds):
            idx = i * len_fold
            off = len(values) if i == self.n_folds - 1 else idx + len_fold
            ts, ss, vs = targets[idx:off], sources[idx:off], values[idx:off]

            h_e = tf.gather(unsq_x, ss)
            t_e = tf.gather(unsq_x, ts)
            r_e = tf.gather(r, vs)
            w = tf.gather(self.rel_kernel, vs)
            h_e = tf.squeeze(tf.matmul(h_e, w), axis=1)
            t_e = tf.squeeze(tf.matmul(t_e, w), axis=1)
            pi = tf.reduce_sum(tf.multiply(t_e, tf.tanh(h_e + r_e)), axis=1)
            pis.append(pi)
        pis = tf.concat(pis, axis=0)

        # Build yet another sparse matrix for storing attention values
        attn_coef = tf.sparse.SparseTensor(
            indices=indices,
            values=pis,
            dense_shape=self.adj_shape
        )

        # Apply the softmax on the neighbours-related attentions
        attn_coef = tf.sparse.softmax(attn_coef)

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

        if self.return_attn_coef:
            return output, attn_coef
        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
        }
