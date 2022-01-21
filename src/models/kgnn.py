import numpy as np
import tensorflow as tf

from scipy import sparse
from keras import models, layers, regularizers, callbacks

from utilities.math import convert_to_tensor
from layers.kgat_conv import KGATConv
from layers.reduction import ReductionLayer


class KGAT(models.Model):
    def __init__(
        self,
        adj_matrix,
        n_layers=2,
        embedding_dim=8,
        final_node="concatenation",
        dropout=None,
        l2_regularizer=None,
        n_folds=32,
        **kwargs
    ):
        if not sparse.issparse(adj_matrix):
            raise ValueError("The adjacency matrix must be sparse for KGAT")
        super().__init__()

        # Initialize the adjacency matrix constant parameter
        self.adj_matrix = convert_to_tensor(adj_matrix, dtype=tf.int32)

        self.n_folds = n_folds
        self.att_adj_matrix = tf.sparse.eye(
            adj_matrix.shape[0], adj_matrix.shape[1],
            dtype=tf.float32
        )

        # Instantiate the regularizer
        if l2_regularizer is not None:
            regularizer = regularizers.l2(l2_regularizer)
        else:
            regularizer = None

        # Initialize the entities embedding weights
        self.embeddings = self.add_weight(
            name='ent_embeddings',
            shape=(adj_matrix.shape[0], embedding_dim),
            initializer='glorot_uniform',
            regularizer=regularizer
        )

        # Initialize the relation types embedding weights
        n_relations = len(np.unique(adj_matrix.shape))
        self.rel_embeddings = self.add_weight(
            name='rel_embeddings',
            shape=(n_relations, embedding_dim),
            initializer='glorot_uniform',
            regularizer=regularizer
        )

        # Initialize the relation types kernel weights
        self.rel_kernels = self.add_weight(
            name="rel_kernels",
            shape=[n_relations, embedding_dim, embedding_dim],
            initializer='glorot_uniform',
            regularizer=regularizer
        )

        # Build the KGAT convolutional layers
        self.kgnn_layers = [
            KGATConv(
                embedding_dim,
                activation='leaky_relu',
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer
            )
            for _ in range(n_layers)
        ]

        # Build the dropout layer
        self.dropout = layers.Dropout(dropout) if dropout else None

        # Build the reduction layer
        self.reduce = ReductionLayer(final_node)

    def call(self, inputs, **kwargs):
        x = self.embeddings
        hs = [x]
        for gnn in self.kgnn_layers:
            x = gnn([x, self.att_adj_matrix])
            if self.dropout is not None:
                x = self.dropout(x)
            hs.append(x)

        # Reduce the outputs of each GCN layer
        return self.reduce(hs)

    def update_attentive_adjacency(self):
        # The adjacency matrix is assumed to be only one such that
        # the value of a_ij is the relation id of an edge from entity i to entity j
        indices, values = self.adj_matrix.indices, self.adj_matrix.values
        targets, sources = indices[:, 1], indices[:, 0]
        unsq_x = tf.expand_dims(self.embeddings, axis=1)

        # Computes the attention values for all the observed interactions
        pis = []
        len_fold = len(values) // self.n_folds
        for i in range(self.n_folds):
            idx = i * len_fold
            off = len(values) if i == self.n_folds - 1 else idx + len_fold
            ts, ss, vs = targets[idx:off], sources[idx:off], values[idx:off]

            h_e = tf.gather(unsq_x, ss)
            t_e = tf.gather(unsq_x, ts)
            r_e = tf.gather(self.rel_embeddings, vs)
            w = tf.gather(self.rel_kernels, vs)
            h_e = tf.squeeze(tf.matmul(h_e, w), axis=1)
            t_e = tf.squeeze(tf.matmul(t_e, w), axis=1)
            pi = tf.reduce_sum(tf.multiply(t_e, tf.tanh(h_e + r_e)), axis=1)
            pis.append(pi)
        pis = tf.concat(pis, axis=0)

        # Build yet another sparse matrix for storing attention values
        att_adj_matrix = tf.sparse.SparseTensor(
            indices=indices,
            values=pis,
            dense_shape=self.adj_matrix.shape
        )

        # Apply the softmax on the neighbours-related attentions
        self.att_adj_matrix = tf.sparse.softmax(att_adj_matrix)


class KGATCallback(callbacks.Callback):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def set_model(self, model):
        if not isinstance(model.gnn, KGAT):
            raise ValueError("The KGAT callback must be used only when training KGAT GNN layers")
        self.model = model

    def on_train_begin(self, logs=None):
        self.log.info("Computing KGAT attentive adjacency matrix")
        self.model.gnn.update_attentive_adjacency()

    def on_epoch_end(self, epoch, logs=None):
        self.log.info("Computing KGAT attentive adjacency matrix")
        self.model.gnn.update_attentive_adjacency()
