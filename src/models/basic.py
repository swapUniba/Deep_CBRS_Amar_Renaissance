import abc
import tensorflow as tf
from tensorflow.keras import models, layers

from models.dense import build_dense_network, build_dense_classifier
from models.gnn import GCN, GAT, GraphSage, LightGCN, KGAT


class BasicRS(models.Model):
    def __init__(
        self,
        dense_units=(512, 256, 128),
        clf_units=(64, 64),
        activation='relu',
        **kwargs
    ):
        """
        :param dense_units: Dense networks units for the Basic recommender system.
        :param clf_units: Classifier network units for the Basic recommender system.
        :param activation: The activation function to use.
        :param **kwargs: Additional args not used.
        """
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
        dense_units=(32, 16),
        clf_units=(16, 16),
        activation='relu',
        **kwargs
    ):
        """
        Initialize a Basic recommender system based on Graph Neural Networks (GCN).

        :param dense_units: Dense networks units for the Basic recommender system.
        :param clf_units: Classifier network units for the Basic recommender system.
        :param activation: The activation function to use.
        :param **kwargs: Additional args not used.
        """
        super().__init__()

        # Build the Basic recommender system
        self.rs = BasicRS(dense_units, clf_units, activation=activation)

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
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on Graph Convolutional Networks (GCN).
        """
        super().__init__(**kwargs)
        self.gnn = GCN(*args, **kwargs)


class BasicGAT(BasicGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on Graph Attention Networks (GAT).
        """
        super().__init__(**kwargs)
        self.gnn = GAT(*args, **kwargs)


class BasicGraphSage(BasicGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on GraphSage.
        """
        super().__init__(**kwargs)
        self.gnn = GraphSage(*args, **kwargs)


class BasicLightGCN(BasicGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on LightGCN.
        """
        super().__init__(**kwargs)
        self.gnn = LightGCN(*args, **kwargs)


class BasicKGAT(BasicGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on Knowledge Graph Attention Networks (KGAT).
        """
        super().__init__(**kwargs)
        self.gnn = KGAT(*args, **kwargs)
