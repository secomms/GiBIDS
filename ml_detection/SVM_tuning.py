from variables import user_CIC_dirname, gb_dir_name, random_state
from db_CIC import db_CIC0, to_scale

import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

param_grid = {
    'C': [1, 5, 10, 100, 1000, 10000, 100000],
    'gamma': [0.01, 0.1, 0.5, 1],
}


def print_dict(f, dictionary):
    for dict_key, dict_value in dictionary.items():
        f.write(f"{dict_key}:" + '\n')
        f.write(f"{dict_value}:" + '\n')


def print_results(save_path, best_par, best_sc):
    """
    Print to file hyperparameters tuning results

    :param save_path: Absolute file path where results must be written
    :param best_par: Chosen hyperparameters
    :param best_sc: 5CV F1-Score for such hyperparameters
    :return:
    """
    with open(save_path, 'w') as f:
        f.write('Hyperparameters tuning results \n')
        f.write('Evaulated hyperparams: \n')
        print_dict(f, param_grid)
        f.write('Selected features: ' + str(features))
        f.write('\n\n----------------\n')
        f.write('Chosen hyperparams: ' + str(best_par) + '\n')
        f.write('F1 5CV: ' + str(best_sc) + '\n')
        f.close()


def evaluate(data: list, n_jobs: int):
    """
    Execute the grid search with globally defined grid parameters

    :param data: list containing features training data (pandas.DataFrame)
                    and binary label training data (pandas.Series)
    :param n_jobs: number of parallel jobs for Feature Selection execution
    :return: the best hyperparameters and related 5CV F1-Score
    """
    [X_train, y_train] = data
    clf_grid = GridSearchCV(SVC(kernel='rbf', random_state=random_state), param_grid=param_grid, cv=5, verbose=2,
                            scoring='f1',
                            n_jobs=n_jobs)
    clf_grid.fit(X_train, y_train)
    return clf_grid.best_params_, clf_grid.best_score_


if __name__ == "__main__":
    subdir_name = "/all_days"
    ds = 0  # dataset identification [ 0 -> CIC-IDS2017 ]
    n_jobs = 8
    sigma = 1000
    mode = 0

    if ds == 0:  # CIC-IDS2017
        binary_label = 'Binary_Label'
        multi_label = 'Label'
        dataset_path = (user_CIC_dirname + gb_dir_name + subdir_name + '/sigma_' + str(sigma) + '/mode_' +
                        str(mode) + '/CIC_IDS_s' + str(sigma) + '_mode' + str(mode) + '_train.csv')
        save_results_path = (user_CIC_dirname + gb_dir_name + subdir_name + '/sigma_' + str(sigma) + '/mode_' +
                             str(mode) + '/params_tuned_SVC_s' + str(sigma) + '_mode' + str(mode) + '.txt')

        features = db_CIC0['feats'][int(sigma)][int(mode)]
        print('Loading dataset...')
        dataset = pd.read_csv(dataset_path, low_memory=False)
        X_train = dataset[features]
        y_train = dataset[binary_label]

        to_be_scaled = list(set(features).intersection(set(to_scale)))
        if to_be_scaled:
            scaler = StandardScaler()
            scaler.fit(X_train[to_be_scaled])
            X_train[to_be_scaled] = scaler.transform(X_train[to_be_scaled])

        data = [X_train, y_train]
        print('Tuning hyperparameters...')
        best_params, best_score = evaluate(data, n_jobs)
        print_results(save_results_path, best_params, best_score)
