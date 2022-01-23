import numpy as np
import tensorflow as tf

from scipy import sparse
from tensorflow.keras import models, regularizers

from utilities.math import convert_to_tensor
from layers.kgcn_conv import KGCNConv


class KGCN(models.Model):
    def __init__(
        self,
        n_users,
        adj_matrix,
        n_layers=2,
        embedding_dim=8,
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

        # Initialize the users embedding weights
        self.user_embeddings = self.add_weight(
            name='user_embeddings',
            shape=(n_users, embedding_dim),
            initializer='glorot_uniform',
            regularizer=regularizer
        )

        # Initialize the entity embedding weights
        n_entities = adj_matrix.shape[0]
        self.ent_embeddings = self.add_weight(
            name='ent_embeddings',
            shape=(n_entities, embedding_dim),
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

    def call(self, inputs, **kwargs):
        x = self.ent_embeddings
        for gnn in self.kgnn_layers:
            x = gnn([x, self.rel_embeddings, self.adj_matrix])

        # Concat the user and (updated) entities embeddings
        return tf.concat([self.user_embeddings, x], axis=0)
