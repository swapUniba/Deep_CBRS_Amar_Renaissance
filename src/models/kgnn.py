import numpy as np
import tensorflow as tf

from scipy import sparse
from tensorflow.keras import models, layers, regularizers

from utilities.math import convert_to_tensor
from layers.kgcn_conv import KGCNConv
from layers.reduction import ReductionLayer


class KGCN(models.Model):
    def __init__(
        self,
        adj_matrix,
        n_layers=2,
        embedding_dim=8,
        final_node="concatenation",
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
            shape=(n_relations + 1, embedding_dim),  # +1 because relation IDs start by one
            initializer='glorot_uniform',
            regularizer=regularizer
        )

        # Build the KGCN convolutional layers
        self.kgnn_layers = [
            KGCNConv(
                embedding_dim,
                activation='relu',
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
            x = gnn([x, self.rel_embeddings, self.adj_matrix])
            if self.dropout is not None:
                x = self.dropout(x)
            hs.append(x)

        # Reduce the outputs of each GCN layer
        return self.reduce(hs)
