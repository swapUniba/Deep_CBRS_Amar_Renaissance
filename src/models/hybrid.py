import abc
import tensorflow as tf

from tensorflow.keras import models, layers

from layers.fusion import FusionLayer
from models.dense import build_dense_network, build_dense_classifier, build_residual_dense_network
from models.gnn import GCN, GAT, GraphSage, LightGCN, DGCF
from models.tsgnn import TwoStepGraphSage, TwoStepGAT, TwoStepDGCF, TwoStepGCN, TwoStepLightGCN
from models.twgnn import TwoWayDGCF, TwoWayGAT, TwoWayLightGCN, TwoWayGraphSage, TwoWayGCN


class HybridCBRS(models.Model):
    """
    Hybrid recommender system that receives inputs from two sources
    """
    def __init__(
            self,
            feature_based=True,
            dense_units=((512, 256, 128), (512, 256, 128), (64, 64)),
            clf_units=(64, 64),
            activation='relu',
            fusion_method='concatenate',
            residual=False,
            **kwargs
    ):
        """
        :param feature_based: if True recommendation is based on user features:
            ((UserGraph, ItemGraph), (UserBert, ItemBert))
            otherwise is based on entities:
            ((UserGraph, UserBert), (ItemGraph, ItemBert))
        :param dense_units: Dense networks units for the Hybrid recommender system (for each branch).
        :param clf_units: Classifier network units for the Hybrid recommender system.
        :param activation: The activation function to use.
        :param fusion_method: The method for fusion. It can be either 'concatenate' or 'attention'.
        :param residual: Whether to use residual connections.
        :param **kwargs: Additional args not used.
        """
        super().__init__()
        self.feature_based = feature_based

        # Instantiate the fusion layers
        if feature_based:
            self.fuse1a = FusionLayer('concatenate')
            self.fuse1b = FusionLayer('concatenate')
            self.fuse2 = FusionLayer(fusion_method)
        else:
            self.fuse1a = FusionLayer(fusion_method)
            self.fuse1b = FusionLayer(fusion_method)
            self.fuse2 = FusionLayer('concatenate')

        # Instantiate dense layers
        self.dense1a = build_dense_network(dense_units[0], activation=activation)
        self.dense1b = build_dense_network(dense_units[0], activation=activation)
        self.dense2a = build_dense_network(dense_units[1], activation=activation)
        self.dense2b = build_dense_network(dense_units[1], activation=activation)
        self.dense3a = build_dense_network(dense_units[2], activation=activation)
        self.dense3b = build_dense_network(dense_units[2], activation=activation)

        # Instantiate the classifier network
        if residual:
            if dense_units[2][-1] != clf_units[-1]:
                raise ValueError("The last dense units before the last fusion layer "
                                 "must be equal to the last classifier units for residual connections")
            self.residual = build_residual_dense_network(clf_units, activation=activation)
            self.activation = layers.Activation(activation)
            self.clf = build_dense_classifier([], n_classes=1)
        else:
            self.residual = self.activation = None
            self.clf = build_dense_classifier(clf_units, n_classes=1, activation=activation)

    def call(self, inputs, **kwargs):
        ug, ig, ub, ib = inputs
        ug = self.dense1a(ug)
        ig = self.dense1b(ig)
        ub = self.dense2a(ub)
        ib = self.dense2b(ib)

        if self.feature_based:
            x1 = self.dense3a(self.fuse1a([ug, ig]))
            x2 = self.dense3b(self.fuse1b([ub, ib]))
        else:
            x1 = self.dense3a(self.fuse1a([ug, ub]))
            x2 = self.dense3b(self.fuse1b([ig, ib]))
        x = self.fuse2([x1, x2])

        if self.residual is None:
            return self.clf(x)
        return self.clf(self.activation(self.residual(x) + x1 + x2))


class HybridBertGNN(abc.ABC, models.Model):
    def __init__(
            self,
            dense_units=(32, 16),
            clf_units=(16, 16),
            feature_based=False,
            activation='relu',
            fusion_method='concatenate',
            residual=False,
            **kwargs
    ):
        """
        Initialize an hybrid recommender system based on Graph Neural Networks (GNNs) and BERT embeddings.

        :param feature_based:
        :param dense_units: Dense networks units for the Basic recommender system.
        :param clf_units: Classifier network units for the Basic recommender system.
        :param activation: The activation function to use.
        :param fusion_method: The method for fusion. It can be either 'concatenate' or 'attention'.
        :param residual: Whether to use residual connections.
        :param **kwargs: Additional args not used.
        """
        super().__init__()

        # Build the Basic recommender system
        self.rs = HybridCBRS(
            feature_based=feature_based,
            dense_units=dense_units,
            clf_units=clf_units,
            activation=activation,
            fusion_method=fusion_method,
            residual=residual
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


def BasicGNNFactory(name, Parent, GNN):
    def __init__(self, *args, **kwargs):
        Parent.__init__(self, **kwargs)
        self.gnn = self.gnn_class(*args, **kwargs)

    basic_gnn = type(name, (Parent,), {"gnn_class": GNN, "__init__": __init__})
    return basic_gnn


class HybridBertTSGNN(HybridBertGNN):
    pass


class HybridBertTWGNN(HybridBertGNN):
    pass


HYBRID_GNNS = [
    (HybridBertGNN, [GCN, GAT, GraphSage, LightGCN, DGCF],
     None),
    (HybridBertTSGNN, [TwoStepGCN, TwoStepGraphSage, TwoStepGAT, TwoStepLightGCN, TwoStepDGCF],
     lambda name: 'HybridBertTS' + name[7:]),
    (HybridBertTWGNN, [TwoWayGCN, TwoWayGraphSage, TwoWayGAT, TwoWayLightGCN, TwoWayDGCF],
     lambda name: 'HybridBertTW' + name[6:]),
]


def generate_hybrids():
    for parent, gnns, name_getter in HYBRID_GNNS:
        for gnn in gnns:
            if name_getter is not None:
                name = name_getter(gnn.__name__)
            else:
                name = 'HybridBert' + gnn.__name__
            globals()[name] = BasicGNNFactory(name, parent, gnn)


# Generate gnns when module is loaded
generate_hybrids()
