import tensorflow as tf

from scipy import sparse
from tensorflow.keras import models, regularizers, layers

from spektral.layers.convolutional import GCNConv

from utilities.math import convert_to_tensor
from layers.kgcn_conv import KGCNConv
from layers.reduction import ReductionLayer


class KGCN(models.Model):
    def __init__(
        self,
        n_users,
        n_items,
        adj_matrix,
        n_layers=2,
        embedding_dim=8,
        final_node='concatenation',
        dropout=None,
        l2_regularizer=None,
        **kwargs
    ):
        if not sparse.issparse(adj_matrix[0]) or not sparse.issparse(adj_matrix[1]):
            raise ValueError("The adjacency matrices must be sparse for KGCN")
        super().__init__()

        # Note normalizing the adjacency matrices using the GCN filter
        bi_adj_matrix = GCNConv.preprocess(adj_matrix[0])
        kg_adj_matrix = GCNConv.preprocess(adj_matrix[1])

        # Initialize the adjacency matrices constant parameters
        self.bi_adj_matrix = convert_to_tensor(bi_adj_matrix, dtype=tf.float32)
        self.kg_adj_matrix = convert_to_tensor(kg_adj_matrix, dtype=tf.float32)

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

        # Initialize the items embedding weights
        self.item_embeddings = self.add_weight(
            name='item_embeddings',
            shape=(n_items, embedding_dim),
            initializer='glorot_uniform',
            regularizer=regularizer
        )

        # Initialize the properties embedding weights
        n_props = self.kg_adj_matrix.shape[0] - n_items
        self.prop_embeddings = self.add_weight(
            name='prop_embeddings',
            shape=(n_props, embedding_dim),
            initializer='glorot_uniform',
            regularizer=regularizer
        )

        # Build the KGCN convolutional layers
        self.seq_layers = [
            KGCNConv(
                n_users,
                n_items,
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
        u_e = self.user_embeddings
        i_e = self.item_embeddings
        p_e = self.prop_embeddings
        x = tf.concat([u_e, i_e, p_e], axis=0)
        hs = [x]
        for gnn in self.seq_layers:
            u_e, i_e, p_e = gnn([u_e, i_e, p_e, self.bi_adj_matrix, self.kg_adj_matrix])
            x = tf.concat([u_e, i_e, p_e], axis=0)
            if self.dropout is not None:
                x = self.dropout(x)
            hs.append(x)

        # Reduce the outputs of each GCN layer
        return self.reduce(hs)


