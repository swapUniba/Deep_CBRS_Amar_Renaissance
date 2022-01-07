import abc
import tensorflow as tf
from scipy import sparse
from tensorflow.keras import models, layers, regularizers

from spektral.layers import GATConv, GCNConv, GraphSageConv

from models.basic import BasicRS
from layers.lightgcn_conv import LightGCNConv
from layers.reduction import ReductionLayer
from utilities.math import sparse_matrix_to_tensor, get_ngrade_neighbors


class BasicGNN(abc.ABC, models.Model):
    def __init__(
        self,
        adj_matrix,
        n_hops,
        embedding_dim=8,
        dropout=None,
        l2_regularizer=None,
        final_node="concatenation",
        dense_units=(32, 16),
        clf_units=(16, 16),
        activation='relu',
        cache_neighbours=False,
        **kwargs
    ):
        """
        Initialize a Basic recommender system based on Graph Neural Networks (GCN).

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_hops: Distance from which every node will be convoluted to.
        :param embedding_dim: The dimension of latent features representations of user and items.
        :param dropout: The dropout to apply after each GCN layer. It can be None.
        :param l2_regularizer: L2 factor to apply on embeddings and GCN layers' weights. It can be None.
        :param final_node: Defines how the final node will be represented from layers. One between the following:
                           'concatenation', 'sum', 'mean', 'w-sum', 'last'.
        :param dense_units: Dense networks units for the Basic recommender system.
        :param clf_units: Classifier network units for the Basic recommender system.
        :param activation: The activation function to use.
        :param cache_neighbours: Whether to pre-compute and cache the neighbours of each node. This is useful only
                                 if the adjacency matrix is very sparse and n_hops is relatively small.
        :param **kwargs: Additional args not used.
        """
        self.cache_neighbours = cache_neighbours
        super().__init__()

        # Initialize the L2 regularizer, if specified
        if l2_regularizer is not None:
            self.regularizer = regularizers.l2(l2_regularizer)
        else:
            self.regularizer = None

        # Initialize the nodes embedding weights
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(adj_matrix.shape[0], embedding_dim),
            initializer='glorot_uniform',
            regularizer=self.regularizer
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
            self.n_grade_adjacency = get_ngrade_neighbors(adj_matrix, n_hops)

        # Build GNN layers
        gnn_kwargs = {
            'regularizer': self.regularizer
        }
        self.gnn_layers = [self.build_gnn_layer(i, **gnn_kwargs) for i in range(n_hops)]

        # Build the dropout layer
        self.dropout = layers.Dropout(dropout) if dropout else None

        # Build the reduction layer
        self.reduce = ReductionLayer(final_node)

        # Build the Basic recommender system
        self.rs = BasicRS(dense_units, clf_units, activation=activation)

    @abc.abstractmethod
    def build_gnn_layer(self, i, **kwargs):
        """
        Abstract method that builds the i-th GNN layer.

        :param i: The index.
        :param kwargs: Additional parameters.
        """
        pass

    def call(self, inputs, **kwargs):
        # Compute the hidden states given by each GCN layer
        x = self.embeddings
        hs = [x]
        for gnn in self.gnn_layers:
            x = gnn([x, self.adj_matrix])
            if self.dropout is not None:
                x = self.dropout(x)
            hs.append(x)

        # Reduce the outputs of each GCN layer
        x = self.reduce(hs)
        return self.embed_recommend(x, inputs)

    def embed_recommend(self, embeddings, inputs):
        """
        Lookup for user and item representations and pass through the recommender model
        :param inputs: (user, item)
        :param embeddings: embeddings produced from previous layers
        :return: Recommendation
        """
        u, i = inputs
        u = tf.nn.embedding_lookup(embeddings, u)
        i = tf.nn.embedding_lookup(embeddings, i)
        return self.rs([u, i])


class BasicGCN(BasicGNN):
    def __init__(
            self,
            adj_matrix,
            n_hiddens=(8, 8, 8),
            **kwargs
    ):
        """
        Initialize a Basic recommender system based on Graph Convolutional Networks (GCN).

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_hiddens: A sequence of numbers of hidden units for each GCN layer.
        """
        self.n_hiddens = n_hiddens

        # Note normalizing the adjacency matrix using the GCN filter
        adj_matrix = GCNConv.preprocess(adj_matrix)
        super().__init__(
            adj_matrix,
            len(n_hiddens),
            **kwargs)

    def build_gnn_layer(self, i, regularizer=None, **kwargs):
        return GCNConv(
            self.n_hiddens[i],
            activation='relu',
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer
        )


class BasicGAT(BasicGNN):
    def __init__(
            self,
            adj_matrix,
            n_hiddens=(8, 8, 8),
            dropout_rate=0.2,
            **kwargs
    ):
        """
        Initialize a Basic recommender system based on Graph Attention Networks (GAT).

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_hiddens: A sequence of numbers of hidden units for each GAT layer.
        :param dropout_rate: The dropout rate to apply to the attention coefficients in GAT.
        """
        self.n_hiddens = n_hiddens
        self.dropout_rate = dropout_rate

        super().__init__(
            adj_matrix,
            len(n_hiddens),
            **kwargs)

    def build_gnn_layer(self, i, regularizer=None, **kwargs):
        return GATConv(
            self.n_hiddens[i],
            dropout_rate=self.dropout_rate,
            activation='relu',
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer
        )


class BasicGraphSage(BasicGNN):
    def __init__(
            self,
            adj_matrix,
            n_hiddens=(8, 8, 8),
            aggregate='mean',
            **kwargs
    ):
        """
        Initialize a Basic recommender system based on GraphSage.

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_hiddens: A sequence of numbers of hidden units for each GraphSage layer.
        :param aggregate: Which aggregation function to use in update (mean, max, ...)
        """
        self.n_hiddens = n_hiddens
        self.aggregate = aggregate

        super().__init__(
            adj_matrix,
            len(n_hiddens),
            **kwargs)

    def build_gnn_layer(self, i, regularizer=None, **kwargs):
        return GraphSageConv(
            self.n_hiddens[i],
            activation='relu',
            aggregate=self.aggregate,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer
        )


class BasicLightGCN(BasicGNN):
    def __init__(
            self,
            adj_matrix,
            n_layers=3,
            **kwargs
    ):
        """
        Initialize a Basic recommender system based on LightGCN.

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_layers: The number of sequential LightGCN layers.
        """
        # Override final_node parameter to 'mean'
        kwargs['final_node'] = 'mean'

        # Note normalizing the adjacency matrix using the GCN filter
        adj_matrix = LightGCNConv.preprocess(adj_matrix)
        super().__init__(
            adj_matrix,
            n_layers,
            **kwargs)

    def build_gnn_layer(self, i, **kwargs):
        return LightGCNConv()
