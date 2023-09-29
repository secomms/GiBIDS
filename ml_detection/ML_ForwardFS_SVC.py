# Functions and main for Feature Selection execution.
# Note that it must be applied only to training set.

import pandas as pd
from variables import user_CIC_dirname, gb_dir_name, random_state
from db_CIC import f_only_gb, to_scale
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from datetime import datetime

max_features = 8


def forward_feature_selection_mlxtend(X: pd.DataFrame, y: pd.Series, n_jobs=6):
    """
    Run Forward Feature Selection from mlxtend library

    :param X: Training set features
    :param y: Training set class label column
    :param n_jobs: Number of parallel jobs for Forward Feature Selection execution
    :return: Complex object containing execution results
    """
    svc = SVC(kernel='rbf', verbose=0, random_state=random_state)
    forward_fs = sfs(estimator=svc,
                     k_features=(1, max_features),
                     forward=True, cv=5, scoring='f1',
                     verbose=2, n_jobs=n_jobs)

    selected = forward_fs.fit(X, y)
    return selected


def init(dataset: pd.DataFrame, binary_lab: str, save_results_path: str, n_jobs=6):
    """
    Compute Forward Feature Selection applied only to graph-based feature and print results on file.

    :param dataset: Complete dataset
    :param binary_lab: Label name for binary label
    :param save_results_path: Absolute path for results file to be written
    :param n_jobs: Number of parallel jobs for Feature Selection execution
    :return: None (Results are written inside save_results_path)
    """

    X_train = dataset[f_only_gb]
    y_train = dataset[binary_lab]

    scaler = StandardScaler()
    scaler.fit(X_train[to_scale])
    X_train[to_scale] = scaler.transform(X_train[to_scale])

    fs_model = forward_feature_selection_mlxtend(X_train[f_only_gb], y_train, n_jobs=n_jobs)
    feats_scelte = list(fs_model.k_feature_names_)
    risultati_completi = fs_model.subsets_

    with open(save_results_path, 'w') as f:
        f.write('Selected features: ' + str(feats_scelte) + '\n\n\n')
        f.write('Complete results: ' + '\n')
        for dict_key, dict_value in risultati_completi.items():
            f.write(f"{dict_key}:" + '\n')
            for key, value in dict_value.items():
                f.write(f"  {key}: {value}" + '\n')
        f.close()


if __name__ == "__main__":
    ds = 0
    sigmas = [1000, 5000, 0]
    modes = [0, 1, 2]
    n_jobs = 8
    subdir_name = '/all_days'

    for sigma in sigmas:
        for mode in modes:
            if ds == 0:  # CIC-IDS2017
                binary_label = 'Binary_Label'
                multi_label = 'Label'

                now = datetime.now()
                h = now.hour
                m = now.minute
                print(f"Starting time {h}:{m}")
                name = 'sigma: ' + str(sigma) + ', mode: ' + str(mode)
                dataset_path = (user_CIC_dirname + gb_dir_name + subdir_name + '/sigma_' + str(sigma) + '/mode_' +
                                str(mode) + '/CIC_IDS_s' + str(sigma) + '_mode' + str(mode) + '_train.csv')

                save_results_path = (user_CIC_dirname + gb_dir_name + subdir_name + '/sigma_' + str(sigma) + '/mode_'
                                     + str(mode) + '/feats_sel_SVC_s' + str(sigma) + '_mode' + str(mode) + '.txt')

                dataset = pd.read_csv(dataset_path, low_memory=False)
                init(dataset, binary_label, save_results_path, n_jobs=n_jobs)
