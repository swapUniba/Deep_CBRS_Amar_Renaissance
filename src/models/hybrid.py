from tensorflow import keras


class HybridCBRS(keras.Model):
    def __init__(self, feature_based=False):
        super().__init__()
        self.feature_based = feature_based
        self.dense1a = self.build_dense_block()
        self.dense1b = self.build_dense_block()
        self.dense2a = self.build_dense_block()
        self.dense2b = self.build_dense_block()
        self.concat = keras.layers.Concatenate()
        self.dense3a = self.build_dense_block(64, 32)
        self.dense3b = self.build_dense_block(64, 32)
        self.fc = self.build_dense_classifier()

    @staticmethod
    def build_dense_block(hsize1=256, hsize2=64):
        return keras.Sequential([
            keras.layers.Dense(hsize1, activation='relu'),
            keras.layers.Dense(hsize2, activation='relu')
        ])

    @staticmethod
    def build_dense_classifier(hsize1=32, hsize2=8):
        return keras.Sequential([
            keras.layers.Dense(hsize1, activation='relu'),
            keras.layers.Dense(hsize2, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        ug, ig, ub, ib = inputs
        ugvec = self.dense1a(ug)
        igvec = self.dense1b(ig)
        ubvec = self.dense2a(ub)
        ibvec = self.dense2b(ib)

        if self.feature_based:
            uigvec = self.dense3a(self.concat([ugvec, igvec]))
            uibvec = self.dense3b(self.concat([ubvec, ibvec]))
            return self.fc(self.concat([uigvec, uibvec]))
        else:
            uvec = self.dense3a(self.concat([ugvec, ubvec]))
            ivec = self.dense3b(self.concat([igvec, ibvec]))
            return self.fc(self.concat([uvec, ivec]))
