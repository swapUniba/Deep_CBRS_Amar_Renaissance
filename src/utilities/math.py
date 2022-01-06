import numpy as np
import tensorflow as tf
from scipy import sparse


def sparse_matrix_to_tensor(x, dtype=tf.float32):
    """
    Convert a Scipy sparse matrix to a Tensorflow's SparseTensor.

    :param x: The input Scipy sparse matrix.
    :param dtype: The target data type.
    :return: The equivalent Tensorflow's SparseTensor.
    """
    assert sparse.issparse(x), "The input matrix should be sparse"

    x = x.tocoo()  # Convert to Coordinate (COO) format first
    x = tf.sparse.SparseTensor(  # Initialize the SparseTensor
        indices=np.mat([x.row, x.col]).T,
        values=x.data,
        dense_shape=x.shape
    )
    x = tf.cast(x, dtype)  # Cast to the specified data type

    # Some Tensorflow operations required an ordering of the sparse representation
    return tf.sparse.reorder(x)
