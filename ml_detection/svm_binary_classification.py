# This python file contains functions to execute the classification task:
#   1) Training and Test sets loading
#   2) Retrieving of selected features and SVM-RBF hyperparameters
#   3) Scaling through Standardization
#   4) 10CV of Training Set
#   5) Testing
#   6) Results evaluation

import pandas as pd
from variables import user_CIC_dirname, gb_dir_name, random_state
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from db_CIC import db_CIC0, to_scale
from binary_performance_evaluation import *
from sklearn.model_selection import cross_val_score

binary_label = 'Binary_Label'
multi_label = 'Label'


def start_SVM(sigma, mode, subdir_name='/all_days'):
    """
    Run the binary classification task through SVM-RBF previously tuned

    :param sigma: block size
    :param mode: Weights configuration to be considered
                    mode = 0 -> unweighted graph
                    mode = 1 -> weighted graph
                    mode = 2 -> mix-weighted graph
    :param subdir_name: name of the subdirectory inside the dataset directory containing all configurations data
                        DEFAULT: '/all_days'

    :return: None, results are printed on screen
    """

    X_train, y_train, X_test, y_test, df_test, svc_params = init_SVM_binary(sigma, mode, subdir_name)
    classification_task(X_train, y_train, X_test, y_test, svc_params, df_test)


def init_SVM_binary(sigma, mode, subdir_name='/all_days'):
    """
    Initialize binary classification of CIC-IDS2017 dataset with SVM-RBF classifier in terms of features to be used and
    training and test set loading
    :param sigma: block size
    :param mode: Weights configuration to be considered
                    mode = 0 -> unweighted graph
                    mode = 1 -> weighted graph
                    mode = 2 -> mix-weighted graph
    :param subdir_name: name of the subdirectory inside the dataset directory containing all configurations data
                        DEFAULT: '/all_days'
    :return:
    """

    features = db_CIC0['feats'][int(sigma)][int(mode)]
    svc_params = db_CIC0['params'][int(sigma)][int(mode)]
    print("Loading data... ")
    df_train = pd.read_csv(user_CIC_dirname + gb_dir_name + subdir_name + '/sigma_' + str(sigma) + '/mode_' + str(
        mode) + '/CIC_IDS_s' + str(sigma) + '_mode' + str(mode) + '_train.csv')
    df_test = pd.read_csv(user_CIC_dirname + gb_dir_name + subdir_name + '/sigma_' + str(sigma) + '/mode_' + str(
        mode) + '/CIC_IDS_s' + str(sigma) + '_mode' + str(mode) + '_test.csv')
    X_train = df_train[features].copy()
    y_train = df_train[binary_label]
    X_test = df_test[features].copy()
    y_test = df_test[binary_label]
    to_be_scaled = list(set(features).intersection(set(to_scale)))
    if to_be_scaled:
        scaler = StandardScaler()
        scaler.fit(X_train[to_be_scaled])
        X_train[to_be_scaled] = scaler.transform(X_train[to_be_scaled])
        X_test[to_be_scaled] = scaler.transform(X_test[to_be_scaled])
    return X_train, y_train, X_test, y_test, df_test, svc_params


def classification_task(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                        svc_params: dict, df_test: pd.DataFrame):

    """
    Having loaded and scaled the training and the test sets, classification task is executed
    :param X_train: -
    :param y_train: -
    :param X_test:  -
    :param y_test:  -
    :param svc_params: hyperparameters stored in db_CIC0
    :param df_test: dataset of test set containing also the multi-class column in order to show not only binary class
                    errors, but also specific attack types misclassified
    :return: None, results are printed on screen
    """

    model = SVC(C=svc_params['C'], gamma=svc_params['gamma'], random_state=random_state)
    print("10CV Training... ")

    f1s = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10, scoring='f1', n_jobs=-1, verbose=0)
    print('F1-Score 10CV: ' + str(f1s))
    print("F1-Score avg: " + str(np.mean(f1s)))
    print("F1-Score std. dev. : " + str(np.std(f1s)))

    model = SVC(C=svc_params['C'], gamma=svc_params['gamma'], random_state=random_state)
    print("Training...")
    model.fit(X_train, y_train)
    print("Testing... ")
    y_pred = model.predict(X_test)
    print("Suport Vectors: " + str(model.n_support_))
    print()
    print("ML Metrics")
    print(get_metrics(y_test, y_pred))
    FPR, FNR, AUC = get_other_metrics(y_true=y_test, y_pred=y_pred)
    print("FPR = " + str(FPR))
    print("FNR = " + str(FNR))
    print("AUC = " + str(AUC))
    print()
    print("CLASSIFICATION ERRORS: ")
    print(get_classification_errors(y_pred, df_test, binary_label, multi_label))
