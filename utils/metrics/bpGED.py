import torch
from scipy.optimize import linear_sum_assignment


def bp_ged(g1, g2, cost_ins=1, cost_delete=1):
    """
    Calculate the DBGE matrix. Y is GED while y_real is the graph's label

    Args:
        g1 (Graph-NetworkX): NetworkX's graph
        g2 (Graph-NetworkX): NetworkX's graph
        cost_ins (int): node insertion cost
        cost_delete (int): node deletion cost

    Returns:
        total_cost: GED approximation using BP-GED algorithm between g1 and g2
    """
    # Cloning input graphs to avoid modifying original data
    a = g1.clone()
    b = g2.clone()

    # Get number of nodes
    n_node_a = a.x.shape[0]
    n_node_b = b.x.shape[0]

    # Initialize cost matrices
    matrix_1 = torch.zeros((n_node_a, n_node_b))
    matrix_2 = torch.eye(n_node_a)
    matrix_3 = torch.eye(n_node_b)
    matrix_4 = torch.zeros((n_node_b, n_node_a))

    # Compute the edges
    _, counts_a = torch.unique(a.edge_index[0], return_counts=True)
    _, counts_b = torch.unique(b.edge_index[0], return_counts=True)
    ext_a = torch.zeros(n_node_a)
    ext_b = torch.zeros(n_node_b)
    ext_a[:counts_a.shape[0]] = counts_a
    ext_b[:counts_b.shape[0]] = counts_b

    # Compute costs for adding or removing edges
    edge_remove = ext_a + cost_delete
    edge_add = ext_b + cost_ins
    edge_add = edge_add[:n_node_b]

    # Compute absolute differences in edge counts between nodes
    ext_a = ext_a.unsqueeze(0)
    ext_b = ext_b.unsqueeze(1)
    edge_diff = torch.abs(ext_a - ext_b).T
    edge_diff = edge_diff[:, :n_node_b]

    # Update cost matrix with edge operation costs
    matrix_1 += edge_diff
    matrix_2 = matrix_2 * edge_remove
    matrix_3 = matrix_3 * edge_add

    # Compute node-wise differences in feature vectors
    matrix_a = a.x.unsqueeze(1).expand(-1, b.x.shape[0], -1)
    matrix_b = b.x.unsqueeze(0).expand(a.x.shape[0], -1, -1)
    matrix_node = torch.sum(torch.abs(matrix_a - matrix_b), -1)
    matrix_node += matrix_1

    # Complete cost matrix
    matrix_2[matrix_2 == 0] = float("Inf")
    matrix_3[matrix_3 == 0] = float("Inf")
    matrix_node = torch.cat((matrix_node, matrix_2), -1)
    matrix_down = torch.cat((matrix_3, matrix_4), -1)
    matrix_cost = torch.cat((matrix_node, matrix_down), 0)

    # Solve the linear sum assignment problem
    row_ind, col_ind = linear_sum_assignment(matrix_cost)

    # Cost of the optimal alignment
    total_cost = matrix_cost[row_ind, col_ind].sum().item()

    return total_cost
