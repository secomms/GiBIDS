# Main for dataset augmentation with graph-based features as proposed
from variables import user_CIC_dirname, gb_dir_name
import pandas as pd
from project_utils import load_and_etl_CIC_IDS_2017
from project_utils import make_directory
from gb_features_gen import init


src_label = 'Source_IP'  # Dataset column name for source IP
dst_label = 'Destination_IP'  # Dataset column name for destination IP
save_path = user_CIC_dirname + gb_dir_name

sigma = 0
mode = 0
verbose = True

if __name__ == "__main__":
    days = load_and_etl_CIC_IDS_2017(user_CIC_dirname)
    make_directory(save_path)
    save_path = save_path + '/all_days'
    make_directory(save_path)

    dataset = pd.DataFrame()
    for day in days:
        dataset = pd.concat([dataset, day])

    print('Files will be saved in path: ' + save_path + '/sigma_' + str(sigma) + '/mode_' + str(mode))
    make_directory(save_path + '/sigma_' + str(sigma))
    make_directory(save_path + '/sigma_' + str(sigma) + '/mode_' + str(mode))

    init(dataset, src_label, dst_label, sigma,
         save_path=save_path + '/sigma_' + str(sigma) + '/mode_' + str(mode) + '/CIC_IDS',
         mode=mode,
         verbose=verbose)

