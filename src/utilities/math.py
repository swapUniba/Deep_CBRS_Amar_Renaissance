import numpy as np
import tensorflow as tf
from scipy import sparse


def convert_to_tensor(x, dtype=tf.float32):
    if sparse.issparse(x):
        return sparse_matrix_to_tensor(x, dtype=dtype)
    return tf.convert_to_tensor(x, dtype=dtype)


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


def get_ngrade_neighbors(adjacency_matrix, grade):
    """
    Builds an adjacency matrix connecting neighbors of "grade" hops

    :param adjacency_matrix: a graph adjacency matrix
    :param grade: distance we want to connect each node to another in the resulting adjacency matrix
    :return: adjacency matrix of grade "grade"
    """
    adj = tf.cast(tf.not_equal(adjacency_matrix, 0), tf.int32)
    result = tf.Variable(adj)

    i = tf.constant(0)
    condition = lambda i: tf.less(i, grade - 1)

    def body(i):
        result.assign(tf.matmul(result, adj, a_is_sparse=True, b_is_sparse=True))
        return [tf.add(i, 1)]

    _ = tf.while_loop(condition, body, [i])

    return tf.cast(tf.not_equal(result, 0), tf.int32)


def get_sub_adjacency_matrix(inputs, n_grade_adjacency):
    """
    Temp method to test
    """
    u, i = inputs
    indices = tf.concat([u, i], 0)
    items = tf.gather(n_grade_adjacency, indices)
    neigh = tf.reduce_sum(items, axis=0)
    tf.where(neigh == 1)