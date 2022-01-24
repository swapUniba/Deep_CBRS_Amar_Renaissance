import abc
import tensorflow as tf
from tensorflow.keras import models, layers

from models.dense import build_dense_network, build_dense_classifier
from models.gnn import GCN, GAT, GraphSage, LightGCN, DGCF
from models.kgnn import KGCN
from models.tsgnn import TwoStepGCN, TwoStepGraphSage, TwoStepGAT, TwoStepLightGCN, TwoStepDGCF
from models.twgnn import TwoWayGCN, TwoWayGraphSage, TwoWayGAT, TwoWayLightGCN, TwoWayDGCF


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


class BasicKGCN(BasicGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on Knowledge Graph Convolutional Networks (KGCNs).
        """
        super().__init__(**kwargs)
        self.gnn = KGCN(*args, **kwargs)


class BasicDGCF(BasicGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on DGCF.
        """
        super().__init__(**kwargs)
        self.gnn = DGCF(*args, **kwargs)


class BasicTSGNN(BasicGNN):
    pass


class BasicTWGNN(BasicGNN):
    pass

class BasicTSGCN(BasicTSGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on TwoStep Graph Convolutional Networks (KGCNs).
        """
        super().__init__(**kwargs)
        self.gnn = TwoStepGCN(*args, **kwargs)


class BasicTSGAT(BasicTSGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on TwoStep Graph Convolutional Networks (KGCNs).
        """
        super().__init__(**kwargs)
        self.gnn = TwoStepGAT(*args, **kwargs)


class BasicTSGraphSage(BasicTSGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on TwoStep Graph Convolutional Networks (KGCNs).
        """
        super().__init__(**kwargs)
        self.gnn = TwoStepGraphSage(*args, **kwargs)


class BasicTSLightGCN(BasicTSGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on TwoStep LightGCN.
        """
        super().__init__(**kwargs)
        self.gnn = TwoStepLightGCN(*args, **kwargs)


class BasicTSDGCF(BasicTSGNN):
    def __init__(self, *args, **kwargs):
        """
        Initialize a Basic recommender system based on TwoStep DGCF.
        """
        super().__init__(**kwargs)
        self.gnn = TwoStepDGCF(*args, **kwargs)


def BasicGNNFactory(name, Parent, GNN):
    def __init__(self, *args, **kwargs):
        Parent.__init__(self, **kwargs)
        self.gnn = self.gnn_class(*args, **kwargs)

    basic_gnn = type(name, (Parent,), {"gnn_class": GNN, "__init__": __init__})
    return basic_gnn


BASIC_GNNS = [
    # (BasicGNN, [GCN, GAT, GraphSage, LightGCN, DGCF]),
    # (BasicGNN, [KGCN]),
    # (BasicTSGNN, [TwoStepGCN, TwoStepGraphSage, TwoStepGAT, TwoStepLightGCN, TwoStepDGCF]),
    (BasicTWGNN, [TwoWayGCN, TwoWayGraphSage, TwoWayGAT, TwoWayLightGCN, TwoWayDGCF],
     lambda name: 'BasicTW' + name[6:]),
]


def generate_basics():
    for parent, gnns, name_getter in BASIC_GNNS:
        for gnn in gnns:
            if name_getter is not None:
                name = name_getter(gnn.__name__)
            else:
                name = 'Basic' + gnn.__name__
            globals()[name] = BasicGNNFactory(name, parent, gnn)


# Generate gnns when module is loaded
generate_basics()
