from tensorflow import keras


class BasicRS(keras.Model):
    def __init__(self, dense_units=(512, 256, 128), fc_units=(64, 64), activation='relu'):
        super().__init__()
        self.concat = keras.layers.Concatenate()
        self.unet = self.build_dense_block(dense_units, activation=activation)
        self.inet = self.build_dense_block(dense_units, activation=activation)
        self.fc = self.build_dense_classifier(fc_units, activation=activation)

    @staticmethod
    def build_dense_block(dense_units, activation='relu'):
        return keras.Sequential([
            keras.layers.Dense(units, activation=activation)
            for units in dense_units
        ])

    @staticmethod
    def build_dense_classifier(fc_units, activation='relu'):
        return keras.Sequential([
            keras.layers.Dense(units, activation=activation)
            for units in fc_units
        ] + [
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        u, i = inputs
        u = self.unet(u)
        i = self.inet(i)
        x = self.concat([u, i])
        y = self.fc(x)
        return y
