from spektral.layers import ops
from spektral.layers.convolutional.conv import Conv
from spektral.utils import gcn_filter


class DGCFConv(Conv):
    r"""
    A Deoscillated Adaptive Graph Collaborative Filtering (LightGCN) from the paper

    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(

        )

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def call(self, inputs, mask=None):
        pass

    @staticmethod
    def preprocess(a):
        return gcn_filter(a)
