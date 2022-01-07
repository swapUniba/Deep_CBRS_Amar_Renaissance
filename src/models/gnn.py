import tensorflow as tf
from scipy import sparse
from tensorflow.keras import models, layers

from layers.reduction import ReductionLayer
from utilities.math import sparse_matrix_to_tensor, get_ngrade_neighbors


class SequentialGNN(models.Model):
    def __init__(
        self,
        adj_matrix,
        seq_layers,
        embedding_dim=8,
        final_node='concatenation',
        dropout=None,
        regularizer=None,
        cache_neighbours=False
    ):
        """
        Initialize a sequence of Graph Neural Networks (GNNs) layers.

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param seq_layers: A list of GNN layers.
        :param n_hops: Distance from which every node will be convoluted to.
        :param embedding_dim: The dimension of latent features representations of user and items.
        :param final_node: Defines how the final node will be represented from layers. One between the following:
                           'concatenation', 'sum', 'mean', 'w-sum', 'last'.
        :param dropout: The dropout to apply after each GCN layer. It can be None.
        :param regularizer: The regularizer object to use for the embeddings. It can be None.
        :param cache_neighbours: Whether to pre-compute and cache the neighbours of each node. This is useful only
                                 if the adjacency matrix is very sparse and n_hops is relatively small.
        """
        super().__init__()
        self.cache_neighbours = cache_neighbours

        # Initialize the nodes embedding weights
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(adj_matrix.shape[0], embedding_dim),
            initializer='glorot_uniform',
            regularizer=regularizer
        )

        # Initialize the adjacency matrix constant parameter and n grade neighbors matrix
        if sparse.issparse(adj_matrix):
            self.adj_matrix = sparse_matrix_to_tensor(adj_matrix, dtype=tf.float32)
        else:
            self.adj_matrix = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)

        # Compute the n-grade adjacency matrix, if needed
        if self.cache_neighbours:
            if sparse.issparse(adj_matrix):
                adj_matrix = adj_matrix.todense()
            self.n_grade_adjacency = get_ngrade_neighbors(adj_matrix, self.n_hops)

        # Build the dropout layer
        self.dropout = layers.Dropout(dropout) if dropout else None

        # Build the reduction layer
        self.reduce = ReductionLayer(final_node)

        # Build GNN layers
        self.seq_layers = seq_layers

    @property
    def n_hops(self):
        return len(self.seq_layers)

    def __len__(self):
        return self.n_hops

    def call(self, inputs, **kwargs):
        # Compute the hidden states given by each GCN layer
        x = self.embeddings
        hs = [x]
        for gnn in self.seq_layers:
            x = gnn([x, self.adj_matrix])
            if self.dropout is not None:
                x = self.dropout(x)
            hs.append(x)

        # Reduce the outputs of each GCN layer
        return self.reduce(hs)
