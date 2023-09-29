# Functions for graph-based features extraction from a given graph

import multiprocessing
import pandas as pd
from graph_tool import clustering
import graph_tool.centrality as gtc
import graph_tool.all as gt


def compute_graph_features(connections_list: pd.DataFrame, G: gt.Graph, mode: int, pool: multiprocessing.Pool):
    """
    Compute graph-based features for each participant of each connection contained inside the block processed.

    :param pool: Threads-pool
    :param connections_list: set of connections in the block cast to Pandas Dataframe for processing reasons.
                             Each row = [src_id, dst_id, edge weight]
    :param G: Graph populated until the last block (G)
    :param mode: Weights configuration to be considered
                    mode = 0 -> unweighted graph
                    mode = 1 -> weighted graph
                    mode = 2 -> mix-weighted graph

    :return: Computed features
    """

    max_distance = 2
    weights_G = None if mode == 0 else G.edge_properties["weights"]
    H = G.copy()
    H.set_directed(False)
    weights_H = None if mode == 0 or mode == 2 else H.edge_properties["weights"]
    nodes = list(set(connections_list['src_id']).union(set(connections_list['dst_id'])))

    dict_degrees = pool.apply_async(get_in_out_degrees, args=(nodes, G, weights_G))
    c_coeff = pool.apply_async(get_clustering_coeff, args=(H, max_distance))
    betweenness = pool.apply_async(get_betweenness, args=(H, weights_H))
    closeness = pool.apply_async(get_closeness, args=(H, weights_H))
    eigenvector_p_map = pool.apply_async(get_eigenvector, args=(H, weights_H))

    return betweenness.get(), c_coeff.get(), closeness.get(), eigenvector_p_map.get(), dict_degrees.get()


def get_eigenvector(H, weights_H):
    _, eigenvector_p_map = gtc.eigenvector(H, epsilon=1.0e-3, weight=weights_H, max_iter=1000)  # O(V * qualcosa)
    return eigenvector_p_map


def get_closeness(H, weights_H):
    closeness = gtc.closeness(H, weight=weights_H, norm=True)
    return closeness


def get_clustering_coeff(H, max_depth):
    c_coeff = clustering.extended_clustering(H, undirected=True, max_depth=max_depth)
    return c_coeff


def get_pagerank(H, weights_H):
    page_rank = gtc.pagerank(H, weight=weights_H)
    return page_rank


def get_betweenness(H, weights_H):
    betweenness, _ = gtc.betweenness(H, weight=weights_H, norm=True)
    return betweenness


def get_in_out_degrees(nodes, G, weights_G):
    degrees = G.get_total_degrees(nodes, eweight=weights_G)
    in_degrees = G.get_in_degrees(nodes, eweight=weights_G)
    out_degrees = G.get_out_degrees(nodes, eweight=weights_G)

    dict_degrees = {}
    for i in range(len(nodes)):
        dict_degrees[nodes[i]] = [degrees[i], in_degrees[i], out_degrees[i]]
    return dict_degrees


