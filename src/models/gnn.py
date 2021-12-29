from spektral.layers import GATConv
from tensorflow.keras.models import Model, Sequential


class BasicGAT(Model):
    def __init__(self, n_hiddens: iter, dropouts):
        """
        :param n_hiddens: list of number of hidden units. For each one a GAT layer will be stacked
        :param dropouts: list of dropouts value. Must match the length of n_hiddens
        """
        super().__init__()
        self.gat_stack = Sequential([
            GATConv(n_hidden, dropout_rate=dropout)
            for n_hidden, dropout in zip(n_hiddens, dropouts)
        ])

    def call(self, inputs):
        out = self.gat_stack(inputs)
        return out
