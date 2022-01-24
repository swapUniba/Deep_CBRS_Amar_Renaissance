import numpy as np

from scipy import sparse


def get_user_properties(ui_adj, ip_adj, n_users, n_items):
    n_properties = ip_adj.shape[0] - n_items

    ui_rows, ui_cols = ui_adj.row, ui_adj.col
    ip_rows, ip_cols = ip_adj.row, ip_adj.col
    ip_rows += n_users
    ip_cols += n_users
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
