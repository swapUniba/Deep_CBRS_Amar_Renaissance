import numpy as np
from tensorflow.keras import utils


class UserItemEmbeddings(utils.Sequence):
    def __init__(
        self,
        ratings,
        embeddings,
        batch_size=512,
        shuffle=False,
        seed=42
    ):
        """
        Initialize a sequence of User-Item embeddings.

        :param ratings: A numpy array of triples (UserID, ItemID, Rating).
        :param embeddings: A numpy array that maps UserID and ItemID to embeddings.
        :param batch_size: The batch size.
        :param shuffle: Whether to shuffle the sequence.
        :param seed: The seed value used to shuffle the sequence.
        """
        super().__init__()

        # Set the ratings and the embeddings
        self.ratings = ratings
        self.embeddings = embeddings

        # Set other settings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indexes = None
        self.random_state = None
        self.on_epoch_end()

    def get_users(self):
        return np.unique(self.ratings[:, 0])

    def get_items(self):
        return np.unique(self.ratings[:, 1])

    def __len__(self):
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return int(np.ceil(len(self.ratings) / self.batch_size))

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item embeddings and the ratings.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item embeddings and the ratings.
        """
        batch_idx = idx * self.batch_size
        batch_off = min(batch_idx + self.batch_size, len(self.ratings))
        if self.shuffle:
            ratings = self.ratings[self.indexes[batch_idx:batch_off]]
        else:
            ratings = self.ratings[batch_idx:batch_off]
        user_embeddings = self.embeddings[ratings[:, 0]]
        item_embeddings = self.embeddings[ratings[:, 1]]
        return (user_embeddings, item_embeddings), ratings[:, 2]

    def on_epoch_end(self):
        """
        Shuffles the indexes at the end of every epoch.
        """
        if self.shuffle:
            if self.random_state is None:
                self.random_state = np.random.RandomState(self.seed)
            self.indexes = np.arange(len(self.ratings))
            self.random_state.shuffle(self.indexes)


class HybridUserItemEmbeddings(utils.Sequence):
    def __init__(
        self,
        ratings,
        graph_embeddings,
        bert_embeddings,
        batch_size=512,
        shuffle=False,
        seed=42
    ):
        """
        Initialize a sequence of Hybrid (Graph+BERT) User-Item embeddings.

        :param ratings: A numpy array of triples (UserID, ItemID, Rating).
        :param graph_embeddings: A numpy array that maps UserID and ItemID to Graph embeddings.
        :param bert_embeddings: A numpy array that maps UserID and ItemID to BERT embeddings.
        :param batch_size: The batch size.
        :param shuffle: Whether to shuffle the sequence.
        :param seed: The seed value used to shuffle the sequence.
        """
        # Set the ratings
        self.ratings = ratings

        # Initialize both Graph and BERT embeddings sequences
        self.graph_embeddings = UserItemEmbeddings(
            ratings, graph_embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )
        self.bert_embeddings = UserItemEmbeddings(
            ratings, bert_embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )

    def __len__(self):
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return len(self.graph_embeddings)

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item Graph+BERT embeddings and the ratings.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item Graph+BERT embeddings and the ratings.
        """
        (user_graph_embeddings, item_graph_embeddings), ratings = self.graph_embeddings[idx]
        (user_bert_embeddings, item_bert_embeddings), _ = self.bert_embeddings[idx]
        return (user_graph_embeddings, item_graph_embeddings, user_bert_embeddings, item_bert_embeddings), ratings

    def on_epoch_end(self):
        """
        Calls on_epoch_end() to any sub-sequence.
        """
        self.graph_embeddings.on_epoch_end()
        self.bert_embeddings.on_epoch_end()


class UserItemGraph(utils.Sequence):
    def __init__(
        self,
        ratings,
        adj_matrix,
        batch_size=512,
        shuffle=False,
        seed=42
    ):
        """
        Initialize a sequence of Graph User-Item IDs.

        :param ratings: A numpy array of triples (UserID, ItemID, Rating).
        :param adj_matrix: The adjacency matrix.
        :param batch_size: The batch size.
        :param shuffle: Whether to shuffle the sequence.
        :param seed: The seed value used to shuffle the sequence.
        """
        super().__init__()

        # Set the ratings and the adjacency matrix
        self.ratings = ratings
        self.adj_matrix = adj_matrix

        # Set other settings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indexes = None
        self.random_state = None
        self.on_epoch_end()

    def get_users(self):
        return np.unique(self.ratings[:, 0])

    def get_items(self):
        return np.unique(self.ratings[:, 1])

    def __len__(self):
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return int(np.ceil(len(self.ratings) / self.batch_size))

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item IDs and the ratings.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item IDs and the ratings.
        """
        batch_idx = idx * self.batch_size
        batch_off = min(batch_idx + self.batch_size, len(self.ratings))
        if self.shuffle:
            ratings = self.ratings[self.indexes[batch_idx:batch_off]]
        else:
            ratings = self.ratings[batch_idx:batch_off]
        return (ratings[:, 0], ratings[:, 1]), ratings[:, 2]

    def on_epoch_end(self):
        """
        Shuffles the indexes at the end of every epoch.
        """
        if self.shuffle:
            if self.random_state is None:
                self.random_state = np.random.RandomState(self.seed)
            self.indexes = np.arange(len(self.ratings))
            self.random_state.shuffle(self.indexes)


class UserItemGraphEmbeddings(utils.Sequence):
    def __init__(
        self,
        ratings,
        adj_matrix,
        embeddings,
        batch_size=512,
        shuffle=False,
        seed=42,
    ):
        """
        Initialize a sequence of Graph User-Item IDs and embeddings (e.g. BERT embeddings).

        :param ratings: A numpy array of triples (UserID, ItemID, Rating).
        :param adj_matrix: The adjacency matrix.
        :param embeddings: A numpy array that maps UserID and ItemID to embeddings.
        :param batch_size: The batch size.
        :param shuffle: Whether to shuffle the sequence.
        :param seed: The seed value used to shuffle the sequence.
        """
        super().__init__()

        # Set the ratings and the adjacency matrix
        self.ratings = ratings
        self.adj_matrix = adj_matrix

        # Initialize both Graph and embeddings sequences
        self.graph_ids = UserItemGraph(
            ratings, adj_matrix,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )
        self.embeddings = UserItemEmbeddings(
            ratings, embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )

    def __len__(self):
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return len(self.graph_ids)

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item IDs, the associated embeddings, and the ratings.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item IDs, the associated embeddings, and the ratings.
        """
        (user_ids, item_ids), ratings = self.graph_ids[idx]
        (user_embeddings, item_embeddings), _ = self.embeddings[idx]
        return (user_ids, item_ids, user_embeddings, item_embeddings), ratings

    def on_epoch_end(self):
        """
        Calls on_epoch_end() to any sub-sequence.
        """
        self.graph_ids.on_epoch_end()
        self.embeddings.on_epoch_end()
