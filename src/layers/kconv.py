import warnings

import tensorflow as tf
from tensorflow.keras.layers import Layer

from spektral.utils.keras import (
    deserialize_kwarg,
    is_keras_kwarg,
    is_layer_kwarg,
    serialize_kwarg,
)
from spektral.layers.convolutional.conv import check_dtypes_decorator


class KConv(Layer):
    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.supports_masking = True
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)
        self.call = check_dtypes_decorator(self.call)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        return {**base_config, **keras_config, **self.config}

    @property
    def config(self):
        return {}

    @staticmethod
    def preprocess(a):
        return a


def check_dtypes(inputs):
    if len(inputs) == 3:
        x, r, a = inputs
    else:
        return inputs

    if a.dtype not in (tf.int32, tf.int64):
        warnings.warn(
            f"The adjacency matrix of dtype {a.dtype} is incompatible with the dtype "
            f"of the KConv layers, and has been automatically cast to "
            f"tf.int32."
        )
        a = tf.cast(a, tf.int32)

    output = [_ for _ in [x, r, a] if _ is not None]
    return output
