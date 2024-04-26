import numpy as np
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow

from data_io.fixed import to_fixed, SCALE

"""
Extends https://github.com/joshlk/k-means-constrained.git
"""


def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def mincostflow(n_X, n_C, distances, minCluster=None, maxCluster=None, maxDist=None, fixed=True, max_dist_mode="hard"):
    if not fixed:
        maxDist = int(maxDist * SCALE)

    if max_dist_mode == "soft":
        edges, costs, capacities, supplies = graph_soft(n_X, n_C, distances, minCluster, maxCluster, maxDist, fixed)
    else:
        edges, costs, capacities, supplies = graph_default(n_X, n_C, distances, minCluster, maxCluster, fixed)
        if max_dist_mode == "hard":
            capacities[costs > maxDist] = 0

    try:
        flows = solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X)
    except Exception as e:
        print(f"Failed to solve min cost flow problem:")
        raise e

    labels = np.argmax(flows, axis=1)
    # if flow[label] == 0, then it is an outlier, so we will assign it to n_C
    labels[flows[np.arange(n_X), labels] == 0] = n_C
    labels = np.expand_dims(labels, axis=1)
    labels = labels.astype(np.int32)
    return labels, 0


def graph_soft(n_X, n_C, cost_matrix, size_min, size_max, max_cost, fixed=True):
    # Create indices of nodes
    X_ix = np.arange(n_X)
    C_dummy_ix = np.arange(X_ix[-1] + 1, X_ix[-1] + 1 + n_C)
    C_ix = np.arange(C_dummy_ix[-1] + 1, C_dummy_ix[-1] + 1 + n_C)
    art_ix = C_ix[-1] + 1
    extra_ix = art_ix + 1

    # Edges
    edges_X_C_dummy = cartesian([X_ix, C_dummy_ix])  # All X's connect to all C dummy nodes (C')
    edges_C_dummy_C = np.stack([C_dummy_ix, C_ix], axis=1)  # Each C' connects to a corresponding C (centroid)
    edges_C_art = np.stack([C_ix, art_ix * np.ones(n_C)], axis=1)  # All C connect to artificial node
    edges_X_extra = np.stack([X_ix, extra_ix * np.ones(n_X)], axis=1)  # all X's connect to the redundancy point
    edge_extra_art = np.array([[extra_ix, art_ix]])
    edges = np.concatenate([edges_X_C_dummy, edges_C_dummy_C, edges_C_art, edges_X_extra, edge_extra_art])

    # Costs
    costs_X_C_dummy = cost_matrix.reshape(cost_matrix.size)
    # costs = np.concatenate([costs_X_C_dummy, np.zeros(edges.shape[0] - len(costs_X_C_dummy))])
    costs = np.concatenate([costs_X_C_dummy,
                            np.zeros(edges_C_dummy_C.shape[0]),
                            np.zeros(edges_C_art.shape[0]),
                            np.ones(edges_X_extra.shape[0]) * max_cost,  # Escape preference edges
                            [0]])

    # Capacities - can set for max-k
    capacities_C_dummy_C = size_max * np.ones(n_C)
    # cap_non = T * n_X  # The total supply and therefore wont restrict flow
    cap_non = n_X  # The total supply and therefore wont restrict flow
    capacities = np.concatenate([
        np.ones(edges_X_C_dummy.shape[0]),
        capacities_C_dummy_C,
        cap_non * np.ones(n_C),
        # np.ones(n_X) * (T - 1),
        np.ones(n_X),
        [cap_non]
    ])

    # Sources and sinks
    # supplies_X = np.ones(n_X) * T
    supplies_X = np.ones(n_X)
    supplies_C = -1 * size_min * np.ones(n_C)  # Demand node
    # supplies_art = -1 * (T * n_X - n_C * size_min)  # Demand node
    supplies_art = -1 * (n_X - n_C * size_min)  # Demand node
    supplies = np.concatenate([
        supplies_X,
        np.zeros(n_C),  # C_dummies
        supplies_C,
        [supplies_art],
        [0]
    ])

    # All arrays must be of int dtype for `SimpleMinCostFlow`
    edges = edges.astype('int32')
    if not fixed:
        costs = to_fixed(costs)
    costs = costs.astype('int32')
    capacities = capacities.astype('int32')
    supplies = supplies.astype('int32')

    return edges, costs, capacities, supplies


def graph_default(n_X, n_C, D, size_min, size_max, fixed=True):
    # Create indices of nodes
    X_ix = np.arange(n_X)
    C_dummy_ix = np.arange(X_ix[-1] + 1, X_ix[-1] + 1 + n_C)
    C_ix = np.arange(C_dummy_ix[-1] + 1, C_dummy_ix[-1] + 1 + n_C)
    art_ix = C_ix[-1] + 1

    # Edges
    edges_X_C_dummy = cartesian([X_ix, C_dummy_ix])  # All X's connect to all C dummy nodes (C')
    edges_C_dummy_C = np.stack([C_dummy_ix, C_ix], axis=1)  # Each C' connects to a corresponding C (centroid)
    edges_C_art = np.stack([C_ix, art_ix * np.ones(n_C)], axis=1)  # All C connect to artificial node

    edges = np.concatenate([edges_X_C_dummy, edges_C_dummy_C, edges_C_art])

    # Costs
    costs_X_C_dummy = D.reshape(D.size)
    costs = np.concatenate([costs_X_C_dummy, np.zeros(edges.shape[0] - len(costs_X_C_dummy))])

    # Capacities - can set for max-k
    capacities_C_dummy_C = size_max * np.ones(n_C)
    cap_non = n_X  # The total supply and therefore wont restrict flow
    capacities = np.concatenate([
        np.ones(edges_X_C_dummy.shape[0]),
        capacities_C_dummy_C,
        cap_non * np.ones(n_C)
    ])

    # Sources and sinks
    supplies_X = np.ones(n_X)
    supplies_C = -1 * size_min * np.ones(n_C)  # Demand node
    supplies_art = -1 * (n_X - n_C * size_min)  # Demand node
    supplies = np.concatenate([
        supplies_X,
        np.zeros(n_C),  # C_dummies
        supplies_C,
        [supplies_art]
    ])

    # All arrays must be of int dtype for `SimpleMinCostFlow`
    edges = edges.astype('int32')
    if not fixed:
        costs = to_fixed(costs)
    costs = costs.astype('int32')
    capacities = capacities.astype('int32')
    supplies = supplies.astype('int32')

    return edges, costs, capacities, supplies


def solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X):
    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = SimpleMinCostFlow()

    if (edges.dtype != 'int32') or (costs.dtype != 'int32') \
            or (capacities.dtype != 'int32') or (supplies.dtype != 'int32'):
        raise ValueError("`edges`, `costs`, `capacities`, `supplies` must all be int dtype")

    # Add each edge with associated capacities and cost
    min_cost_flow.add_arcs_with_capacity_and_unit_cost(edges[:, 0], edges[:, 1], capacities, costs)

    # Add node supplies
    for count, supply in enumerate(supplies):
        min_cost_flow.set_node_supply(count, supply)

    status = min_cost_flow.solve()
    # Find the minimum cost flow between node 0 and node 4.
    if status != min_cost_flow.OPTIMAL:
        raise Exception(status)

    # Assignment
    flows = np.array([min_cost_flow.flow(i) for i in range(n_X * n_C)]).reshape(n_X, n_C).astype('int32')

    return flows
