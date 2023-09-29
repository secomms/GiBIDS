# Functions for dataset augmentation with Graph-Based features

import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
from graph_builder import build_graph, init_general_G
from features_extractor import compute_graph_features


def init(dataset: pd.DataFrame, src_label_name, dst_label_name, sigma: int, save_path: str, mode: int, verbose: bool):
    """
    Start execution of proposed approach

    :param dataset: full dataset pre-processed and chronologically ordered (D)
    :param src_label_name: Source IP column name
    :param dst_label_name: Destination IP column name
    :param sigma: block size
    :param save_path: Absolute path reaching the directory where to save the augmented dataset
    :param mode: int
                    mode = 0 -> unweighted graph
                    mode = 1 -> weighted graph
                    mode = 2 -> mix-weighted graph
    :param verbose: show processing infos
    :return: None, augmented dataset is printed to .csv file
    """

    blocks = []
    if sigma > 0:
        for i in range(0, len(dataset), sigma):
            if i + sigma <= len(dataset):
                b = dataset[[src_label_name, dst_label_name]].iloc[i:i + sigma]
            else:
                b = dataset[[src_label_name, dst_label_name]].iloc[i:]
            blocks.append(b)
    else:
        blocks.append(dataset)

    features = process_dataset(blocks, src_label_name, dst_label_name, mode, verbose)
    final_dataset = add_features(dataset, features)
    print("Writing file") if verbose else None
    final_dataset.to_csv(save_path + '_s' + str(sigma) + '_mode' + str(mode) + '.csv')


def add_features(dataset: pd.DataFrame, features: [(str, list)]):
    """
    Add to dataset a column for each graph-based computed feature

    :param dataset: Original dataset
    :param features: Graph-based features computed from original dataset
    :return: Augmented dataset
    """
    for (name, feature_vector) in features:
        dataset[name] = feature_vector
    return dataset


def process_dataset(Blocks: list, src_label_name: str, dst_label_name: str, mode: int, verbose: bool):
    """
    Process given blocks of connections in terms of graph population and features extraction

    :param Blocks: blocks obtained by splitting complete dataset
    :param src_label_name: source-ip column name
    :param dst_label_name: destination-ip column name
    :param mode: Weights configuration to be considered
                    mode = 0 -> unweighted graph
                    mode = 1 -> weighted graph
                    mode = 2 -> mix-weighted graph

    :param verbose: show processing infos
    :return: List of lists containing graph-based features computed for the entire dataset
    """

    n_parallel_measures = 5
    general_G = init_general_G()
    l_degree_src, l_degree_dst, l_in_deg_src, l_in_deg_dst, l_out_deg_src, l_out_deg_dst, \
        l_close_src, l_close_dst, \
        l_betw_src, l_betw_dst, \
        l_eigen_src, l_eigen_dst, \
        l_c_coeff_d1_src, l_c_coeff_d2_src, l_c_coeff_d1_dst, l_c_coeff_d2_dst, \
        l_weights = ([] for _ in range(17))

    with tqdm(total=len(Blocks)) as pbar:
        pbar.set_description('processed blocks')
        threads_pool = multiprocessing.Pool(processes=n_parallel_measures)

        for df in Blocks:
            connections_list = df[[src_label_name, dst_label_name]].copy()
            df_freq = df[[src_label_name, dst_label_name]]
            df_freq = df_freq.assign(temp_freq=1)
            connections_list['freq'] = df_freq.groupby([src_label_name, dst_label_name], sort=False)[
                'temp_freq'].transform('sum')

            print("Populating graph with block connections") if verbose else None
            general_G, src_ID, dst_ID, weights = build_graph(connections_list, general_G, mode=mode, verbose=verbose)
            connections_list['src_id'] = src_ID
            connections_list['dst_id'] = dst_ID
            l_weights = l_weights + weights

            print("Extracting graph-based features for block") if verbose else None

            betweenness, c_coeff, \
                closeness, eigenvector_p_map, dict_degrees = compute_graph_features(connections_list, general_G,
                                                                                    mode, threads_pool)

            c_coeff_d1 = c_coeff[0]
            c_coeff_d2 = c_coeff[1]
            for row in connections_list.itertuples():
                src_id = row[4]
                dst_id = row[5]

                # Source Features
                l_degree_src.append(dict_degrees[src_id][0])
                l_in_deg_src.append(dict_degrees[src_id][1])
                l_out_deg_src.append(dict_degrees[src_id][2])
                l_betw_src.append(betweenness[src_id] if not np.isnan(betweenness[src_id]) else -10)
                l_c_coeff_d1_src.append(c_coeff_d1[src_id])
                l_c_coeff_d2_src.append(c_coeff_d2[src_id])
                l_close_src.append(closeness[src_id] if not np.isnan(closeness[src_id]) else -10)
                l_eigen_src.append(
                    eigenvector_p_map[src_id] if not np.isnan(eigenvector_p_map[src_id]) else -10)

                # Destination Features
                l_degree_dst.append(dict_degrees[dst_id][0])
                l_in_deg_dst.append(dict_degrees[dst_id][1])
                l_out_deg_dst.append(dict_degrees[dst_id][2])
                l_betw_dst.append(betweenness[dst_id] if not np.isnan(betweenness[dst_id]) else -10)
                l_c_coeff_d1_dst.append(c_coeff_d1[dst_id])
                l_c_coeff_d2_dst.append(c_coeff_d2[dst_id])
                l_close_dst.append(closeness[dst_id] if not np.isnan(closeness[dst_id]) else -10)
                l_eigen_dst.append(
                    eigenvector_p_map[dst_id] if not np.isnan(eigenvector_p_map[dst_id]) else -10)

            pbar.update(1)
        threads_pool.close()
        threads_pool.join()

    features = [
        ('src_deg', l_degree_src),
        ('src_in_deg', l_in_deg_src),
        ('src_out_deg', l_out_deg_src),
        ('src_closeness', l_close_src),
        ('src_betweenness', l_betw_src),
        ('src_eigen', l_eigen_src),
        ('src_c_coeff_d1', l_c_coeff_d1_src),
        ('src_c_coeff_d2', l_c_coeff_d2_src),

        ('dst_deg', l_degree_dst),
        ('dst_in_deg', l_in_deg_dst),
        ('dst_out_deg', l_out_deg_dst),
        ('dst_closeness', l_close_dst),
        ('dst_betweenness', l_betw_dst),
        ('dst_eigen', l_eigen_dst),
        ('dst_c_coeff_d1', l_c_coeff_d1_dst),
        ('dst_c_coeff_d2', l_c_coeff_d2_dst),
        ('weights', l_weights)
    ]
    return features
