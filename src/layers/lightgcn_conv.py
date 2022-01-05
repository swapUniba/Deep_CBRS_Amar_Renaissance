from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.utils import gcn_filter


class LightGCNConv(Conv):
    r"""
    A light graph convolutional layer (LightGCN) from the paper

    > [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)<br>
    > Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang

    **Mode**: single, disjoint, mixed, batch.

    This layer computes:
    $$
        \X' = \hat \D^{-1/2} \hat \A \hat \D^{-1/2} \X
    $$
    where \( \hat \A = \A + \I \) is the adjacency matrix with added self-loops
    and \(\hat\D\) is its degree matrix.

    **Input**

    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Modified Laplacian of shape `([batch], n_nodes, n_nodes)`; can be computed with
    `spektral.utils.convolution.gcn_filter`.

    **Output**

    - Node features with the same shape as the input.

    **Arguments**

    - `activity_regularizer`: regularization applied to the output;
    """

    def __init__(
        self,
        activity_regularizer=None,
        **kwargs
    ):
        super().__init__(
            activity_regularizer=activity_regularizer,
            **kwargs
        )

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def call(self, inputs, mask=None):
        x, a = inputs
        output = ops.modal_dot(a, x)
        return output

    @staticmethod
    def preprocess(a):
        return gcn_filter(a)
