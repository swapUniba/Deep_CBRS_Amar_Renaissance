import numpy as np
import tensorflow as tf
from scipy import sparse
from tensorflow.keras import models, layers, regularizers

from spektral.layers import GATConv, GCNConv, GraphSageConv

from models.basic import BasicRS
from layers.lightgcn_conv import LightGCNConv
from layers.reduction import ReductionLayer


class BasicGNN(models.Model):
    def __init__(
        self,
        adj_matrix,
        embedding_dim=8,
        dropout=None,
        l2_regularizer=None,
        final_node="concatenation",
        dense_units=(32, 16),
        clf_units=(16, 16),
        activation='relu',
        **kwargs
    ):
        """
        Initialize a Basic recommender system based on Graph Neural Networks (GCN).

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param embedding_dim: The dimension of latent features representations of user and items.
        :param dropout: The dropout to apply after each GCN layer. It can be None.
        :param l2_regularizer: L2 factor to apply on embeddings and GCN layers' weights. It can be None.
        :param final_node: Defines how the final node will be represented from layers. One between the following:
                           'concatenation', 'sum', 'mean', 'w-sum', 'last'.
        :param dense_units: Dense networks units for the Basic recommender system.
        :param clf_units: Classifier network units for the Basic recommender system.
        :param activation: The activation function to use.
        :param **kwargs: Additional args not used.
        """
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

        # Initialize the adjacency matrix constant parameter
        if sparse.issparse(adj_matrix):
            adj_matrix = adj_matrix.tocoo()
            self.adj_matrix = tf.sparse.reorder(tf.sparse.SparseTensor(
                indices=np.mat([adj_matrix.row, adj_matrix.col]).T,
                values=adj_matrix.data.astype(np.float32, copy=False),
                dense_shape=adj_matrix.shape
            ))
        else:
            self.adj_matrix = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)

        # Build the dropout layer
        self.dropout = layers.Dropout(dropout) if dropout else None

        # Build the reduction layer
        self.reduce = ReductionLayer(final_node)

        # Build the Basic recommender system
        self.rs = BasicRS(dense_units, clf_units, activation=activation)

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
        # Note normalizing the adjacency matrix using the GCN filter
        adj_matrix = GCNConv.preprocess(adj_matrix.astype(np.float32, copy=False))
        super().__init__(
            adj_matrix,
            **kwargs)

        # Build GCN layers
        self.gnn_layers = [
            GCNConv(
                n_hidden,
                activation='relu',  # Or we should just follow LightGCN and set this to None ?
                kernel_regularizer=self.regularizer,
                bias_regularizer=self.regularizer
            )
            for n_hidden in n_hiddens
        ]


class BasicGAT(BasicGNN):
    def __init__(
            self,
            adj_matrix,
            n_hiddens=(8, 8, 8),
            **kwargs
    ):
        """
        Initialize a Basic recommender system based on Graph Attention Networks (GAT).

        :param adj_matrix: The graph adjacency matrix. It can be either sparse or dense.
        :param n_hiddens: A sequence of numbers of hidden units for each GAT layer.
        """
        super().__init__(
            adj_matrix,
            **kwargs)

        # Build GAT layers
        self.gnn_layers = [
            GATConv(
                n_hidden,
                activation='relu',
                kernel_regularizer=self.regularizer,
                bias_regularizer=self.regularizer
            )
            for n_hidden in n_hiddens
        ]
        # Layer concatenation for output
        self.concat = layers.Concatenate()


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
        super().__init__(
            adj_matrix,
            **kwargs)

        # Build GraphSage layers
        self.gnn_layers = [
            GraphSageConv(
                n_hidden,
                activation='relu',
                aggregate=aggregate,
                kernel_regularizer=self.regularizer,
                bias_regularizer=self.regularizer
            )
            for n_hidden in n_hiddens
        ]


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
        adj_matrix = LightGCNConv.preprocess(adj_matrix.astype(np.float32, copy=False))
        super().__init__(
            adj_matrix,
            **kwargs)

        # Build LightGCN layers
        self.gnn_layers = [
            LightGCNConv()
            for _ in range(n_layers)
        ]
