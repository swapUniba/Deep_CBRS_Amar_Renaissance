import abc
import tensorflow as tf

from spektral.layers import GCNConv, GraphSageConv, GATConv
from tensorflow.keras import models, regularizers

from layers.dgcf_conv import DGCFConv
from layers.lightgcn_conv import LightGCNConv
from models.gnn import SequentialGNN, FullInputSequentialGNN


class TwoWayGNN(abc.ABC, models.Model):
    def __init__(
        self,
        n_users,
        n_items,
        adj_matrices,
        n_hops,
        embedding_dim=8,
        user_item_node="mean",
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

        if len(adj_matrices) != 3:
            raise ValueError('Exactly three adjacency matrix are needed!')
        adj_ui_matrix, adj_ip_matrix, adj_up_matrix = adj_matrices

        # Build the first sequential GNN model
        gnn_kwargs = {'regularizer': regularizer}
        way_one_gnn_layers = [self.build_gnn_layer(i, **gnn_kwargs) for i in range(n_hops)]
        self.way_one_gnn_layers = SequentialGNN(
            adj_up_matrix, way_one_gnn_layers,
            embedding_dim=embedding_dim, final_node=user_item_node,
            dropout=dropout, regularizer=regularizer, cache_neighbours=cache_neighbours
        )
        way_two_gnn_layers = [self.build_gnn_layer(i, **gnn_kwargs) for i in range(n_hops)]
        self.way_two_gnn_layers = SequentialGNN(
            adj_ip_matrix, way_two_gnn_layers,
            embedding_dim=embedding_dim, final_node=user_item_node,
            dropout=dropout, regularizer=regularizer, cache_neighbours=cache_neighbours
        )

        # Get the slice of item and users embeddings
        self.n_items = n_items
        self.n_users = n_users

        # Build the second sequential model
        # Get the number of hiddens for the second GNN
        if hasattr(self, 'n_hiddens'):
            if n_hops == len(self.n_hiddens):
                if user_item_node == 'concatenation':
                    second_embedding_dim = embedding_dim * (n_hops + 1)
                    self.n_hiddens.extend([second_embedding_dim for _ in range(n_hops)])
                else:
                    self.n_hiddens.extend([embedding_dim for _ in range(n_hops)])
        step_two_gnn_layers = [self.build_gnn_layer(i + n_hops, **gnn_kwargs) for i in range(n_hops)]
        self.step_two_gnn_layers = FullInputSequentialGNN(
            adj_ui_matrix, step_two_gnn_layers,
            final_node=final_node, dropout=dropout, cache_neighbours=cache_neighbours
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
        users = self.way_one_gnn_layers(None)
        items = self.way_two_gnn_layers(None)
        embeddings = tf.concat([
            tf.slice(users, begin=[0, 0], size=[self.n_users, users.shape[1]]),
            tf.slice(items, begin=[0, 0], size=[self.n_items, items.shape[1]]),
        ], axis=0)
        return self.step_two_gnn_layers(embeddings)


class TwoWayGCN(TwoWayGNN):
    def __init__(
            self,
            n_users,
            n_items,
            adj_matrices,
            n_hiddens=(8, 8, 8),
            **kwargs
    ):
        """
        Initialize a Basic recommender system based on Two Way Graph Convolutional Networks (GCN).

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_hiddens: A sequence of numbers of hidden units for each GCN layer.
        """
        self.n_hiddens = n_hiddens

        # Note normalizing the adjacency matrix using the GCN filter
        adj_matrices = [GCNConv.preprocess(matrix) for matrix in adj_matrices]
        super().__init__(
            n_users,
            n_items,
            adj_matrices,
            len(n_hiddens),
            **kwargs)

    def build_gnn_layer(self, i, regularizer=None, **kwargs):
        return GCNConv(
            self.n_hiddens[i],
            activation='relu',
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer
        )


class TwoWayGraphSage(TwoWayGNN):
    def __init__(
            self,
            n_users,
            n_items,
            adj_matrices,
            n_hiddens=(8, 8, 8),
            aggregate='mean',
            **kwargs
    ):
        """
        Initialize TwoStepGraphSage.

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_hiddens: A sequence of numbers of hidden units for each GraphSage layer.
        :param aggregate: Which aggregation function to use in update (mean, max, ...)
        """
        self.n_hiddens = n_hiddens
        self.aggregate = aggregate

        super().__init__(
            n_users,
            n_items,
            adj_matrices,
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


class TwoWayGAT(TwoWayGNN):
    def __init__(
            self,
            n_users,
            n_items,
            adj_matrix,
            n_hiddens=(8, 8, 8),
            dropout_rate=0.0,
            **kwargs
    ):
        """
        Initialize a TwoStep Graph Attention Networks (GAT).

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_hiddens: A sequence of numbers of hidden units for each GAT layer.
        :param dropout_rate: The dropout rate to apply to the attention coefficients in GAT.
        """
        self.n_hiddens = n_hiddens
        self.dropout_rate = dropout_rate

        super().__init__(
            n_users,
            n_items,
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


class TwoWayLightGCN(TwoWayGNN):
    def __init__(
            self,
            n_users,
            n_items,
            adj_matrix,
            n_layers=3,
            **kwargs
    ):
        """
        Initialize a TwoStep LigthGCN.

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_layers: The number of sequential LightGCN layers.
        """
        # Override final_node parameter to 'mean'
        kwargs['final_node'] = 'mean'

        # Note normalizing the adjacency matrix using the GCN filter
        adj_matrix = [LightGCNConv.preprocess(matrix ) for matrix in adj_matrix]
        super().__init__(
            n_users,
            n_items,
            adj_matrix,
            n_layers,
            **kwargs)

    def build_gnn_layer(self, i, **kwargs):
        return LightGCNConv()


class TwoWayDGCF(TwoWayGNN):
    def __init__(
            self,
            n_users,
            n_items,
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
        crosshop_matrix = [DGCFConv.preprocess(matrix) for matrix in adj_matrix]
        super().__init__(
            n_users,
            n_items,
            crosshop_matrix,
            n_layers,
            **kwargs)

    def build_gnn_layer(self, i, regularizer=None, **kwargs):
        return DGCFConv(regularizer)