import numpy as np
from tensorflow import keras


class UserItemEmbeddings(keras.utils.Sequence):
    def __init__(
        self,
        ratings,
        user_embeddings,
        item_embeddings,
        batch_size=512,
        shuffle=False,
        seed=42
    ):
        """
        A base class for User-Item embeddings sequence.

        :param ratings: A numpy array of triples (UserID, ItemID, Rating).
        :param user_embeddings: A dictionary of UserID and associated embedding.
        :param item_embeddings: A dictionary of ItemID and associated embedding.
        :param batch_size: The batch size.
        :param shuffle: Whether to shuffle the sequence.
        :param seed: The seed value used to shuffle the sequence.
        """
        super().__init__()

        # Set the ratings
        self.ratings = ratings

        # Set the embeddings
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

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
        Get the i-th batch consisting of User-Item embeddings and the rating.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item embeddings and the ratings.
        """
        batch_idx = idx * self.batch_size
        batch_off = min(batch_idx + self.batch_size, len(self.ratings))
        if self.shuffle:
            ratings = self.ratings[self.indexes[batch_idx:batch_off]]
        else:
            ratings = self.ratings[batch_idx:batch_off]
        user_embeddings = self.user_embeddings[ratings[:, 0]]
        item_embeddings = self.item_embeddings[ratings[:, 1]]
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


class HybridUserItemEmbeddings(keras.utils.Sequence):
    def __init__(
        self,
        ratings,
        graph_user_embeddings,
        graph_item_embeddings,
        bert_user_embeddings,
        bert_item_embeddings,
        batch_size=512,
        shuffle=False,
        seed=42
    ):
        self.graph_embeddings = UserItemEmbeddings(
            ratings, graph_user_embeddings, graph_item_embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )
        self.bert_embeddings = UserItemEmbeddings(
            ratings, bert_user_embeddings, bert_item_embeddings,
            batch_size=batch_size, shuffle=shuffle, seed=seed
        )

    @property
    def ratings(self):
        return self.graph_embeddings.ratings

    def __len__(self):
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return len(self.graph_embeddings)

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item embeddings and the rating.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item embeddings and the ratings.
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


class UserItemGraph(keras.utils.Sequence):
    def __init__(
        self,
        ratings,
        adj_matrix,
        batch_size=512,
        shuffle=False,
        seed=42
    ):
        super().__init__()

        # Set the ratings and the adjency matrix
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
        Get the i-th batch consisting of User-Item embeddings and the rating.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item embeddings and the ratings.
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


class UserItemGraphBertEmbeddings(keras.utils.Sequence):
    def __init__(
        self,
        ratings,
        adj_matrix,
        bert_user_embeddings,
        bert_item_embeddings,
        batch_size=512,
        shuffle=False,
        seed=42,
    ):
        super().__init__()

        # Set the ratings and the adjency matrix
        self.ratings = ratings
        self.adj_matrix = adj_matrix
        # Set other settings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indexes = None
        self.random_state = None
        self.on_epoch_end()
        self.user_embeddings = bert_user_embeddings
        self.item_embeddings = bert_item_embeddings

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item IDs and the rating.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item IDs and the ratings.
        """
        batch_idx = idx * self.batch_size
        batch_off = min(batch_idx + self.batch_size, len(self.ratings))
        if self.shuffle:
            ratings = self.ratings[self.indexes[batch_idx:batch_off]]
        else:
            ratings = self.ratings[batch_idx:batch_off]
        user_embeddings = self.user_embeddings[ratings[:, 0]]
        item_embeddings = self.item_embeddings[ratings[:, 1]]
        return (ratings[:, 0], ratings[:, 1]), (user_embeddings, item_embeddings), ratings[:, 2]
