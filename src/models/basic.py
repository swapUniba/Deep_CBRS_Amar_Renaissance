import abc
import tensorflow as tf
from spektral.layers import GCNConv, GATConv, GraphSageConv
from tensorflow.keras import models, layers, regularizers

from layers.lightgcn_conv import LightGCNConv
from models.dense import build_dense_network, build_dense_classifier
from models.gnn import SequentialGNN


class BasicRS(models.Model):
    def __init__(
        self,
        dense_units=(512, 256, 128),
        clf_units=(64, 64),
        activation='relu',
        **kwargs
    ):
        super().__init__()
        self.concat = layers.Concatenate()
        self.unet = build_dense_network(dense_units, activation=activation)
        self.inet = build_dense_network(dense_units, activation=activation)
        self.clf = build_dense_classifier(clf_units, n_classes=1, activation=activation)

    def call(self, inputs, **kwargs):
        u, i = inputs
        u = self.unet(u)
        i = self.inet(i)
        x = self.concat([u, i])
        return self.clf(x)


class BasicGNN(abc.ABC, models.Model):
    def __init__(
        self,
        adj_matrix,
        n_hops,
        embedding_dim=8,
        final_node="concatenation",
        dropout=None,
        l2_regularizer=None,
        cache_neighbours=False,
        dense_units=(32, 16),
        clf_units=(16, 16),
        activation='relu',
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
        :param dense_units: Dense networks units for the Basic recommender system.
        :param clf_units: Classifier network units for the Basic recommender system.
        :param activation: The activation function to use.
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
        self.gnn = SequentialGNN(
            adj_matrix, gnn_layers,
            embedding_dim=embedding_dim, final_node=final_node,
            dropout=dropout, regularizer=regularizer, cache_neighbours=cache_neighbours
        )

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
        updated_embeddings = self.gnn(None)
        return self.embed_recommend(updated_embeddings, inputs)

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
