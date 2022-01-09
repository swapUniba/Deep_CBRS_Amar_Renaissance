import abc
import tensorflow as tf

from tensorflow.keras import models, layers

from models.dense import build_dense_network, build_dense_classifier
from models.gnn import GCN, GAT, GraphSage, LightGCN


class HybridCBRS(models.Model):
    """
    Hybrid recommender system that receives inputs from two sources
    """
    def __init__(
            self,
            feature_based=False,
            dense_units=((512, 256, 128), (64, 64)),
            clf_units=(64, 64),
            activation='relu',
            **kwargs
    ):
        """
        :param feature_based: if True recommendation is based on user features:
            ((UserGraph, ItemGraph), (UserBert, ItemBert))
            otherwise is based on entities:
            ((UserGraph, UserBert), (ItemGraph, ItemBert))
        :param dense_units: Dense networks units for the Basic recommender system.
        :param clf_units: Classifier network units for the Basic recommender system.
        :param activation: The activation function to use.
        :param **kwargs: Additional args not used.
        """
        super().__init__()
        self.feature_based = feature_based
        self.concat = layers.Concatenate()
        self.dense1a = build_dense_network(dense_units[0], activation=activation)
        self.dense1b = build_dense_network(dense_units[0], activation=activation)
        self.dense2a = build_dense_network(dense_units[1], activation=activation)
        self.dense2b = build_dense_network(dense_units[1], activation=activation)
        self.dense3a = build_dense_network(dense_units[2], activation=activation)
        self.dense3b = build_dense_network(dense_units[2], activation=activation)
        self.clf = build_dense_classifier(clf_units, n_classes=1, activation=activation)

    def call(self, inputs, **kwargs):
        ug, ig, ub, ib = inputs
        ug = self.dense1a(ug)
        ig = self.dense1b(ig)
        ub = self.dense2a(ub)
        ib = self.dense2b(ib)

        if self.feature_based:
            x1 = self.dense3a(self.concat([ug, ig]))
            x2 = self.dense3b(self.concat([ub, ib]))
        else:
            x1 = self.dense3a(self.concat([ug, ub]))
            x2 = self.dense3b(self.concat([ig, ib]))
        return self.clf(self.concat([x1, x2]))


class HybridBertGNN(abc.ABC, models.Model):
    def __init__(
            self,
            dense_units=(32, 16),
            clf_units=(16, 16),
            feature_based=False,
            activation='relu',
            **kwargs
    ):
        """
        Initialize an hybrid recommender system based on Graph Neural Networks (GNNs) and BERT embeddings.

        :param feature_based:
        :param dense_units: Dense networks units for the Basic recommender system.
        :param clf_units: Classifier network units for the Basic recommender system.
        :param activation: The activation function to use.
        :param **kwargs: Additional args not used.
        """
        super().__init__()

        # Build the Basic recommender system
        self.rs = HybridCBRS(
            feature_based=feature_based,
            dense_units=dense_units,
            clf_units=clf_units,
            activation=activation,
        )

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
        ug, ig, ub, ib = inputs
        ug = tf.nn.embedding_lookup(embeddings, ug)
        ig = tf.nn.embedding_lookup(embeddings, ig)
        return self.rs([ug, ig, ub, ib])


class HybridBertGCN(HybridBertGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize an hybrid recommender system based on Graph Convolutional Networks (GCN) and BERT embeddings.
        """
        super().__init__(**kwargs)
        self.gnn = GCN(*args, **kwargs)


class HybridBertGAT(HybridBertGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize an hybrid recommender system based on Graph Attention Networks (GAT) and BERT embeddings.
        """
        super().__init__(**kwargs)
        self.gnn = GAT(*args, **kwargs)


class HybridBertGraphSage(HybridBertGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize an hybrid recommender system based on GraphSage and BERT embeddings.
        """
        super().__init__(**kwargs)
        self.gnn = GraphSage(*args, **kwargs)


class HybridBertLightGCN(HybridBertGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize an hybrid recommender system based on GraphSage and BERT embeddings.
        """
        super().__init__(**kwargs)
        self.gnn = LightGCN(*args, **kwargs)
