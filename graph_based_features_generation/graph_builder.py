# Functions for graph population with collected connections

import pandas as pd
from tqdm import tqdm
import graph_tool.all as gt

ids_terminals = {}  # memory used to store integer ids corresponding to IP addresses see


def init_general_G():
    """
    Create an empty direct graph with edge weight property map initialized
    :return: Empty graph
    """
    general_G = gt.Graph(directed=True)
    eweight = general_G.new_ep("double")
    general_G.edge_properties["weights"] = eweight
    return general_G


def current_weight(src: int, dst: int, graph: gt.Graph):
    """
    Get the current weight of an edge linking a source and a destination node

    :param src: id of source
    :param dst: id of destination
    :param graph: Graph at current iteration (Block)
    :return: current weight, edge id
    """
    try:  # edge exists
        edge = graph.edge(src, dst, add_missing=False)
        w0 = graph.edge_properties["weights"][edge]
    except:  # edge do not exist
        w0 = 0
        edge = None

    return w0, edge


def get_id_node_and_add(node: str):
    """
    Uses a dictionary to get the integer id of the given address
    that will be used for finding the corresponding node inside the graph if present.
    If not present, meaning it's seen for the first time, it's added to the dictionary and the id is returned

    :param node: IP address of a node
    :return: integer id for such node
    """
    try:
        id_node = ids_terminals[node]
    except KeyError:
        dict_values = list(ids_terminals.values())
        if not dict_values:
            id_node = 0
            ids_terminals[node] = id_node
        else:
            id_node = max(dict_values) + 1
            ids_terminals[node] = id_node

    return id_node


def build_graph(connections_list: pd.DataFrame, G: gt.Graph, mode: int, verbose: bool):
    """
    Gets the connections inside the block and loads them inside the graph
    considering a variable/constant weight

    :param connections_list: Connections to be loaded on the graph
    :param G: Graph populated until the last processed block B_{j-1}
    :param mode: int
                    mode = 0 -> unweighted graph
                    mode = 1 -> weighted graph
                    mode = 2 -> mix-weighted graph
    :param verbose: show processing infos
    :return: Updated graph, the list of source and destination nodes and list of edge weights
    """

    weights = []
    src_ID = []
    dst_ID = []
    block_memory = {}
    with tqdm(total=len(connections_list), disable=not verbose) as pbar1:
        pbar1.set_description('analysing connections')

        for row in connections_list.itertuples():
            src = row[1]
            dst = row[2]
            in_block_freq = row[3]

            src_id = get_id_node_and_add(src)
            dst_id = get_id_node_and_add(dst)
            key = str(src_id) + '#' + str(dst_id)
            current_w, e = current_weight(src_id, dst_id, G)

            if mode == 0:
                if current_w == 0:  # edge do not exist
                    e = G.add_edge(src_id, dst_id, add_missing=True)
                weight = 1

            elif mode == 1 or mode == 2:
                if current_w == 0:  # edge do not exist
                    e = G.add_edge(src_id, dst_id, add_missing=True)
                    weight = in_block_freq
                    block_memory[key] = weight
                else:
                    if key in block_memory:
                        weight = block_memory[key]
                    else:
                        weight = current_w + in_block_freq
                        block_memory[key] = weight

            G.edge_properties["weights"][e] = weight
            weights.append(weight)  # informative column to be added to the final dataset
                                    # but not intended to be used as a feature
            src_ID.append(src_id)
            dst_ID.append(dst_id)
            pbar1.update(1)
    return G.copy(), src_ID, dst_ID, weights
