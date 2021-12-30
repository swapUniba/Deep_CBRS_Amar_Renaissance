from tensorflow import keras

from models.dense import build_dense_network, build_dense_classifier


class HybridCBRS(keras.Model):
    def __init__(
        self,
        feature_based=False,
        dense_units=((512, 256, 128), (64, 64)),
        clf_units=(64, 64),
        activation='relu'
    ):
        super().__init__()
        self.feature_based = feature_based
        self.concat = keras.layers.Concatenate()
        self.dense1a = build_dense_network(dense_units[0], activation=activation)
        self.dense1b = build_dense_network(dense_units[0], activation=activation)
        self.dense2a = build_dense_network(dense_units[0], activation=activation)
        self.dense2b = build_dense_network(dense_units[0], activation=activation)
        self.dense3a = build_dense_network(dense_units[1], activation=activation)
        self.dense3b = build_dense_network(dense_units[1], activation=activation)
        self.clf = build_dense_classifier(clf_units, n_classes=1, activation=activation)

    def call(self, inputs):
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
