# Variables used during execution

# root directory containing one or multiple datasets directories
user_path = ''
if user_path == '':
    print("user_path variable not defined (variables.py)")
    quit()
# path dataset directory (first level)
user_CIC_dirname = user_path + '/CIC-IDS-2017'

# graph-based directory (second level directory)
gb_dir_name = '/graph_based_feats_gen'

# Random seed to be used for reproduction of ML steps with non-deterministic operations like random sampling
random_state = 46
