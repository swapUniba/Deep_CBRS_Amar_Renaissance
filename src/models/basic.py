from tensorflow import keras


class BasicCBRS(keras.Model):
    def __init__(self):
        super().__init__()
        self.unet = self.build_dense_block()
        self.inet = self.build_dense_block()
        self.concat = keras.layers.Concatenate()
        self.fc = self.build_dense_classifier()

    @staticmethod
    def build_dense_block(hsize1=512, hsize2=256, hsize3=128):
        return keras.Sequential([
            keras.layers.Dense(hsize1, activation='relu'),
            keras.layers.Dense(hsize2, activation='relu'),
            keras.layers.Dense(hsize3, activation='relu')
        ])

    @staticmethod
    def build_dense_classifier(hsize1=64, hsize2=64):
        return keras.Sequential([
            keras.layers.Dense(hsize1, activation='relu'),
            keras.layers.Dense(hsize2, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        u, i = inputs
        u = self.unet(u)
        i = self.inet(i)
        x = self.concat([u, i])
        y = self.fc(x)
        return y
