import numpy as np

from scipy import sparse

from utilities.math import symmetrize_matrix


def get_user_properties(ui_adj, ip_adj, n_users, n_items):
    n_properties = ip_adj.shape[0] - n_items

    ui_rows, ui_cols = ui_adj.row, ui_adj.col
    ip_rows, ip_cols = ip_adj.row, ip_adj.col
    ip_rows = ip_rows + n_users
    ip_cols = ip_cols + n_users
    uip_rows = np.concatenate([ui_rows, ip_rows])
    uip_cols = np.concatenate([ui_cols, ip_cols])
    uip_data = np.concatenate([ui_adj.data, ip_adj.data])
    uip_adj = sparse.coo_matrix(
        (uip_data, (uip_rows, uip_cols)),
        shape=(n_users + n_items + n_properties, n_users + n_items + n_properties)
    )
    # Get crosshop connections

    # Create the user-properties matrix
    uip_adj = uip_adj.dot(uip_adj)
    uip_adj.data = np.ones(len(uip_adj.data))
    uip_adj = uip_adj.todense()
    up_adj = np.zeros(shape=(n_users + n_properties, n_users + n_properties))
    up_adj[n_users:, :n_users] = uip_adj[n_users + n_items:, :n_users]
    up_adj[:n_users, n_users:] = uip_adj[:n_users, n_users + n_items:]
    return sparse.coo_matrix(up_adj)


def build_adjacency_matrix(
        bi_ratings,
        users,
        items,
        props_triples=None,
        props=None,
        type_adjacency='unary',
        sparse_adjacency=True,
        symmetric_adjacency=True
):
    """
    :param bi_ratings: The bipartite ratings as a matrix associating to users and items a 0-1 rating.
    :param users: A sequence of users IDs.
    :param items: A sequence of items IDs.
    :param props_triples: The knowledge graph triples of items and properties. It can be None.
    :param props: A sequence of properties IDs. It can be None.
    :param type_adjacency: Used only if return_adjacency is True and sparse_adjacency is True. It can be either
                           'unary' for 1-only ratings, 'binary' for 0/1-only ratings and 'unary-kg' if you need both the
                           unary matrix and the KG unary graph. In the latter case it requires props and props_triples.
    :param sparse_adjacency: Whether to return the adjacency matrix as a sparse matrix instead of dense.
    :param symmetric_adjacency: Whether to return a symmetric adjacency matrix.
    :return: The adjacency matrix.
    """
    # Unary adjacency matrix (only positive ratings)
    if type_adjacency == 'unary':
        # Compute the dimensions of the adjacency matrix
        adj_size = len(users) + len(items)

        # Instantiate the sparse adjacency matrix
        pos_idx = bi_ratings[:, 2] == 1
        adj_matrix = sparse.coo_matrix(
            (bi_ratings[pos_idx, 2], (bi_ratings[pos_idx, 0], bi_ratings[pos_idx, 1])),
            shape=[adj_size, adj_size], dtype=np.float32
        )

        # Make the matrix symmetric
        if symmetric_adjacency:
            adj_matrix = symmetrize_matrix(adj_matrix)

        # Convert to dense matrix, if specified
        if not sparse_adjacency:
            adj_matrix = adj_matrix.todense()
        return adj_matrix

    # Binary adjacency matrix (both positive and negative ratings)
    if type_adjacency == 'binary':
        # Compute the dimensions of the adjacency matrix
        adj_size = len(users) + len(items)

        # Set the data of the matrix
        coo_data = bi_ratings[:, 2]
        coo_rows, coo_cols = bi_ratings[:, 0], bi_ratings[:, 1]

        # Instantiate the sparse adjacency matrix
        adj_matrix = sparse.coo_matrix(
            (coo_data, (coo_rows, coo_cols)),
            shape=[adj_size, adj_size], dtype=np.float32
        )

        # Make the matrix symmetric
        if symmetric_adjacency:
            adj_matrix = symmetrize_matrix(adj_matrix)

        # Convert to dense matrix, if specified
        if not sparse_adjacency:
            adj_matrix = adj_matrix.todense()
        return adj_matrix

    # Unary KG adjacency matrices (return both unary adjacency matrix and KG adjacency matrix)
    if type_adjacency in ['unary-kg', 'unary-uip']:
        if props is None or props_triples is None:
            raise ValueError("KG adjacency matrix requires properties info")

        # Compute the dimensions of the adjacency matrices
        adj_bi_size = len(users) + len(items)
        adj_kg_size = len(items) + len(props)

        # Instantiate the sparse adjacency matrices
        pos_idx = bi_ratings[:, 2] == 1
        bi_coo_data = bi_ratings[pos_idx, 2]
        bi_coo_rows, bi_coo_cols = bi_ratings[pos_idx, 0], bi_ratings[pos_idx, 1]

        ip_coo_data = props_triples[:, 2]
        ip_coo_rows, ip_coo_cols = props_triples[:, 0], props_triples[:, 1]

        if type_adjacency == 'unary-kg':
            adj_bi_matrix = sparse.coo_matrix(
                (bi_coo_data, (bi_coo_rows, bi_coo_cols)),
                shape=[adj_bi_size, adj_bi_size], dtype=np.float32
            )

            adj_kg_matrix = sparse.coo_matrix(
                (ip_coo_data, (ip_coo_rows, ip_coo_cols)),
                shape=[adj_kg_size, adj_kg_size], dtype=np.float32
            )
            # Make the matrices symmetric
            if symmetric_adjacency:
                adj_bi_matrix = symmetrize_matrix(adj_bi_matrix)
                adj_kg_matrix = symmetrize_matrix(adj_kg_matrix)

            # Convert to dense matrices, if specified
            if not sparse_adjacency:
                adj_bi_matrix = adj_bi_matrix.todense()
                adj_kg_matrix = adj_kg_matrix.todense()
            return adj_bi_matrix, adj_kg_matrix
        else:  # is unary-uip
            ip_coo_rows = ip_coo_rows + len(users)
            ip_coo_cols = ip_coo_cols + len(users)
            uip_coo_data = np.concatenate([bi_coo_data, ip_coo_data])
            uip_coo_rows = np.concatenate([bi_coo_rows, ip_coo_rows])
            uip_coo_cols = np.concatenate([bi_coo_cols, ip_coo_cols])

            adj_uip_matrix = sparse.coo_matrix(
                (uip_coo_data, (uip_coo_rows, uip_coo_cols)),
                shape=[adj_bi_size + len(props), adj_bi_size + len(props)], dtype=np.float32
            )

            # Make the matrices symmetric
            if symmetric_adjacency:
                adj_uip_matrix = symmetrize_matrix(adj_uip_matrix)

            # Convert to dense matrices, if specified
            if not sparse_adjacency:
                adj_uip_matrix = adj_uip_matrix.todense()
            return adj_uip_matrix

    raise ValueError("Unknown adjacency matrix type named {}".format(type_adjacency))