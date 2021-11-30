from tensorflow import keras


class BasicCBRS(keras.Model):
    def __init__(self):
        super().__init__()
        self.unet = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu')
        ])
        self.inet = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu')
        ])
        self.concat = keras.layers.Concatenate()
        self.fc = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        u, i = inputs
        u = self.unet(u)
        i = self.inet(i)
        x = self.concat([u, i])
        y = self.fc(x)
        return y
