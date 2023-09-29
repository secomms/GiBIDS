# Script for training set and test set generation based on binary class balancing of training set.
# The rest of the dataset will be used as test set.

import pandas as pd
from variables import user_CIC_dirname, gb_dir_name, random_state


binary_label = 'Binary_Label'
multi_label = 'Label'
test_perc = 0.7

sigma = 1000
mode = 0
ds = 0

subdir_name = '/all_days'


if __name__ == "__main__":

    if ds == 0:  # CIC-IDS2017
        dataset_path = (user_CIC_dirname + gb_dir_name + subdir_name + '/sigma_' + str(sigma) + '/mode_' + str(mode) +
                        '/CIC_IDS_s' + str(sigma) + '_mode' + str(mode) + '.csv')
        dataset = pd.read_csv(dataset_path)
        train_save_path = (user_CIC_dirname + gb_dir_name + subdir_name + '/sigma_' + str(sigma) + '/mode_' + str(mode) +
                           '/CIC_IDS_s' + str(sigma) + '_mode' + str(mode) + '_train.csv')
        test_save_path = (user_CIC_dirname + gb_dir_name + subdir_name + '/sigma_' + str(sigma) + '/mode_' + str(mode) +
                          '/CIC_IDS_s' + str(sigma) + '_mode' + str(mode) + '_test.csv')

        train_days = 2

        dataset['parsed_ts'] = pd.to_datetime(dataset['Timestamp'], format='%Y-%d-%m %H:%M:%S')
        days = dataset.groupby(dataset['parsed_ts'].dt.date)
        counter = 0
        train = pd.DataFrame()
        df_test = pd.DataFrame()
        for day in days:
            name, data = day
            if counter < train_days:
                train = pd.concat([train, data])
            else:
                df_test = pd.concat([df_test, data])
            counter = counter + 1

        n_malicious = len(train[train['Binary_Label'] == 1])
        df_train = train[train['Binary_Label'] == 1]
        df_train = pd.concat([df_train, train[train['Binary_Label'] == 0].sample(n_malicious,
                                                                                 random_state=random_state)])

        df_train.to_csv(train_save_path)
        df_test.to_csv(test_save_path)
