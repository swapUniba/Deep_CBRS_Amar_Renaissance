from tensorflow import keras


def build_dense_network(units, **kwargs):
    return keras.Sequential([
        keras.layers.Dense(u, **kwargs) for u in units
    ])

def build_dense_classifier(units, n_classes, **kwargs):
    clf_kwargs = kwargs.copy()
    clf_kwargs['activation'] = 'sigmoid' if n_classes == 1 else 'softmax'
    return keras.Sequential([
        keras.layers.Dense(u, **kwargs) for u in units
    ] + [
        keras.layers.Dense(n_classes, **clf_kwargs)
    ])
