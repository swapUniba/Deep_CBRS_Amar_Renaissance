from tensorflow.keras import models, layers


def build_dense_network(units, **kwargs):
    return models.Sequential([
        layers.Dense(u, **kwargs) for u in units
    ])


def build_dense_classifier(units, n_classes, **kwargs):
    clf_kwargs = kwargs.copy()
    clf_kwargs['activation'] = 'sigmoid' if n_classes == 1 else 'softmax'
    return models.Sequential([
        layers.Dense(u, **kwargs) for u in units
    ] + [
        layers.Dense(n_classes, **clf_kwargs)
    ])


def build_residual_dense_network(units, **kwargs):
    last_kwargs = kwargs.copy()
    last_kwargs['activation'] = None
    return models.Sequential([
        layers.Dense(u, **kwargs) for u in units[:-1]
    ] + [
        layers.Dense(units[-1], **last_kwargs)
    ])
