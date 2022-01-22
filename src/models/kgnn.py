import numpy as np
import tensorflow as tf

from scipy import sparse
from tensorflow.keras import models, layers, regularizers

from utilities.math import convert_to_tensor
from layers.kgcn_conv import KGCNConv


class KGCN(models.Model):
    def __init__(
        self,
        adj_matrix,
        n_layers=2,
        embedding_dim=8,
        dropout=None,
        l2_regularizer=None,
        **kwargs
    ):
        if not sparse.issparse(adj_matrix):
            raise ValueError("The adjacency matrix must be sparse for KGCN")
        super().__init__()

        # Initialize the adjacency matrix constant parameter
        self.adj_matrix = convert_to_tensor(adj_matrix, dtype=tf.int64)

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
        n_relations = len(np.unique(adj_matrix.data))
        self.rel_embeddings = self.add_weight(
            name='rel_embeddings',
            shape=(n_relations, embedding_dim),
            initializer='glorot_uniform',
            regularizer=regularizer
        )

        # Build the KGCN convolutional layers
        self.kgnn_layers = [
            KGCNConv(
                embedding_dim,
                activation='relu' if i == n_layers - 1 else 'tanh',
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer
            )
            for i in range(n_layers)
        ]

        # Build the dropout layer
        self.dropout = layers.Dropout(dropout) if dropout else None

    def call(self, inputs, **kwargs):
        x = self.embeddings
        for gnn in self.kgnn_layers:
            x = gnn([x, self.rel_embeddings, self.adj_matrix])
            if self.dropout is not None:
                x = self.dropout(x)
        return x
