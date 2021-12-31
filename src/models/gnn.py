import numpy as np
import tensorflow as tf
from scipy import sparse
from tensorflow import keras

from spektral.utils.convolution import gcn_filter
from spektral.layers import GATConv, GCNConv

from models.basic import BasicRS


class BasicGAT(keras.models.Model):
    def __init__(self, n_hiddens: iter, dropouts):
        """
        :param n_hiddens: list of number of hidden units. For each one a GAT layer will be stacked
        :param dropouts: list of dropouts value. Must match the length of n_hiddens
        """
        super().__init__()
        self.gat_stack = keras.models.Sequential([
            GATConv(n_hidden, dropout_rate=dropout)
            for n_hidden, dropout in zip(n_hiddens, dropouts)
        ])

    def call(self, inputs):
        out = self.gat_stack(inputs)
        return out


class BasicGCN(keras.models.Model):
    def __init__(
        self,
        adj_matrix,
        embedding_dim=8,
        n_hiddens=(8, 8, 8),
        dropout=None,
        l2_regularizer=1e-5,
        dense_units=(32, 16),
        clf_units=(16, 16),
        activation='relu'
    ):
        """
        Initialize a Basic recommender system based on Graph Convolutional Networks (GCN).

        :param adj_matrix: The graph adjency matrix. It can be either sparse or dense.
        :param embedding_dim: The dimension of latent features representations of user and items.
        :param n_hiddens: A sequence of numbers of hidden units for each GCN layer.
        :param dropout: The dropout to apply after each GCN layer. It can be None.
        :param l2_regularizer: L2 regularization constant to apply on embeddings and GCN layers' weights.
        :param dense_units: Dense networks units for the Basic recommender system.
        :param clf_units: Classifier network units for the Basic recommender system.
        :param activation: The activation function to use.
        """
        super().__init__()

        # Initialize the nodes embedding weights
        self.embeddings = self.add_weight(
            shape=(adj_matrix.shape[0], embedding_dim),
            initializer='glorot_uniform',
            regularizer=keras.regularizers.l2(l2_regularizer)
        )

        # Initialize the adjacency matrix constant parameter
        # Note normalizing the adjency matrix using the GCN filter
        adj_matrix = gcn_filter(adj_matrix.astype(np.float32, copy=False))
        if sparse.issparse(adj_matrix):
            adj_matrix = adj_matrix.tocoo()
            self.adj_matrix = tf.sparse.reorder(tf.sparse.SparseTensor(
                indices=np.mat([adj_matrix.row, adj_matrix.col]).T,
                values=adj_matrix.data,
                dense_shape=adj_matrix.shape
            ))
        else:
            self.adj_matrix = tf.convert_to_tensor(adj_matrix)

        # Build GCN layers
        self.gcn_layers = [
            GCNConv(
                n_hidden,
                activation='relu',  # Or we should just follow LightGCN and set this to None ?
                kernel_regularizer=keras.regularizers.l2(l2_regularizer),
                bias_regularizer=keras.regularizers.l2(l2_regularizer)
            )  
            for n_hidden in n_hiddens
        ]

        # Build the dropout layer
        if dropout is not None:
            self.dropout = keras.layers.Dropout(dropout)
        else:
            self.dropout = None

        # Build the concat layer and the Basic recommender system
        self.concat = keras.layers.Concatenate()
        self.rc = BasicRS(dense_units, clf_units, activation=activation)

    def call(self, inputs):
        # Compute the hidden states given by each GCN layer
        x = self.embeddings
        hs = [x]
        for gcn in self.gcn_layers:
            x = gcn([x, self.adj_matrix])
            if self.dropout is not None:
                x = self.dropout(x)
            hs.append(x)

        # Concat the outputs of each GCN layer
        x = self.concat(hs)

        # Lookup for user and item representations and pass through the recommender model
        u, i = inputs
        u = tf.nn.embedding_lookup(x, u)
        i = tf.nn.embedding_lookup(x, i)
        return self.rc((u, i))
