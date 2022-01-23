import abc

import tensorflow as tf
from keras import models, layers, regularizers
from spektral.layers import GCNConv, GraphSageConv, GATConv

from layers.dgcf_conv import DGCFConv
from layers.lightgcn_conv import LightGCNConv
from layers.reduction import ReductionLayer
from utilities.math import convert_to_tensor, get_ngrade_neighbors


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

        # Initialize the adjacency matrix constant parameter
        self.adj_matrix = convert_to_tensor(adj_matrix, dtype=tf.float32)

        # Compute the n-grade adjacency matrix, if needed
        if self.cache_neighbours:
            raise NotImplementedError("Multi-hops neighbours caching is not yet completely supported!")
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
        x = self.embeddings
        hs = [x]
        for gnn in self.seq_layers:
            x = gnn([x, self.adj_matrix])
            if self.dropout is not None:
                x = self.dropout(x)
            hs.append(x)

        # Reduce the outputs of each GCN layer
        return self.reduce(hs)


class InputSequentialGNN(models.Model):
   def __init__(self, adj_matrix, seq_layers, final_node='concatenation', dropout=None,
                cache_neighbours=False, *args, **kwargs):
       """
       Initialize a sequence of Graph Neural Networks (GNNs) layers which embeddings are given in input.

       :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
       :param seq_layers: A list of GNN layers.
       :param final_node: Defines how the final node will be represented from layers. One between the following:
                          'concatenation', 'sum', 'mean', 'w-sum', 'last'.
       :param dropout: The dropout to apply after each GCN layer. It can be None.
       :param regularizer: The regularizer object to use for the embeddings. It can be None.
       :param cache_neighbours: Whether to pre-compute and cache the neighbours of each node. This is useful only
                                if the adjacency matrix is very sparse and n_hops is relatively small.
       """
       super().__init__(*args, **kwargs)
       self.cache_neighbours = cache_neighbours

       # Initialize the adjacency matrix constant parameter
       self.adj_matrix = convert_to_tensor(adj_matrix, dtype=tf.float32)

       # Compute the n-grade adjacency matrix, if needed
       if self.cache_neighbours:
           raise NotImplementedError("Multi-hops neighbours caching is not yet completely supported!")
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
       hs = [inputs]
       for gnn in self.seq_layers:
           inputs = gnn([inputs, self.adj_matrix])
           if self.dropout is not None:
               inputs = self.dropout(inputs)
           hs.append(inputs)

       # Reduce the outputs of each GCN layer
       return self.reduce(hs)


class GNN(abc.ABC, models.Model):
    def __init__(
        self,
        adj_matrix,
        n_hops,
        embedding_dim=8,
        final_node="concatenation",
        dropout=None,
        l2_regularizer=None,
        cache_neighbours=False,
        **kwargs
    ):
        """
        Initialize a Basic recommender system based on Graph Neural Networks (GCN).

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_hops: Distance from which every node will be convoluted to.
        :param embedding_dim: The dimension of latent features representations of user and items.
        :param final_node: Defines how the final node will be represented from layers. One between the following:
                           'concatenation', 'sum', 'mean', 'w-sum', 'last'.
        :param dropout: The dropout to apply after each GCN layer. It can be None.
        :param l2_regularizer: L2 factor to apply on embeddings and GCN layers' weights. It can be None.
        :param cache_neighbours: Whether to pre-compute and cache the neighbours of each node. This is useful only
                                 if the adjacency matrix is very sparse and n_hops is relatively small.
        :param **kwargs: Additional args not used.
        """
        super().__init__()

        # Instantiate the regularizer
        if l2_regularizer is not None:
            regularizer = regularizers.l2(l2_regularizer)
        else:
            regularizer = None

        # Build the sequential GNN model
        gnn_kwargs = {'regularizer': regularizer}
        gnn_layers = [self.build_gnn_layer(i, **gnn_kwargs) for i in range(n_hops)]
        self.gnn_layers = SequentialGNN(
            adj_matrix, gnn_layers,
            embedding_dim=embedding_dim, final_node=final_node,
            dropout=dropout, regularizer=regularizer, cache_neighbours=cache_neighbours
        )

    @abc.abstractmethod
    def build_gnn_layer(self, i, **kwargs):
        """
        Abstract method that builds the i-th GNN layer.

        :param i: The index.
        :param kwargs: Additional parameters.
        """
        pass

    def call(self, inputs, **kwargs):
        return self.gnn_layers(None)


class GCN(GNN):
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


class GAT(GNN):
    def __init__(
            self,
            adj_matrix,
            n_hiddens=(8, 8, 8),
            dropout_rate=0.0,
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


class GraphSage(GNN):
    def __init__(
            self,
            adj_matrix,
            n_hiddens=(8, 8, 8),
            aggregate='mean',
            **kwargs
    ):
        """
        Initialize GraphSage.

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


class LightGCN(GNN):
    def __init__(
            self,
            adj_matrix,
            n_layers=3,
            **kwargs
    ):
        """
        Initialize LightGCN.

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


class DGCF(GNN):
    def __init__(
            self,
            adj_matrix,
            n_layers=3,
            **kwargs
    ):
        """
        Initialize DGCF.

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_layers: The number of sequential DGCF layers.
        """
        # Override final_node parameter to 'mean'
        kwargs['final_node'] = 'mean'

        # Note normalizing the adjacency matrix using the GCN filter and getting the crosshop matrix
        crosshop_matrix = DGCFConv.preprocess(adj_matrix)
        super().__init__(
            crosshop_matrix,
            n_layers,
            **kwargs)

    def build_gnn_layer(self, i, regularizer=None, **kwargs):
        return DGCFConv(regularizer)
