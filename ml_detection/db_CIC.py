# Database of collected infos related to
#   -> graph-based features names
#   -> graph-based features to be scaled through standardization
#   -> features selection results for each configuration of augmented CIC-IDS2017 dataset
#   -> hyperparameters tuning results for each configuration of augmented CIC-IDS2017 dataset
#
# Such infos will be used during ML processes.


f_only_gb = [
    'src_deg', 'src_in_deg', 'src_out_deg', 'src_closeness', 'src_betweenness', 'src_eigen',
    'src_c_coeff_d1', 'src_c_coeff_d2',

    'dst_deg', 'dst_in_deg', 'dst_out_deg', 'dst_closeness', 'dst_betweenness', 'dst_eigen',
    'dst_c_coeff_d1', 'dst_c_coeff_d2'
]

to_scale = ['src_deg', 'src_in_deg', 'src_out_deg', 'dst_deg', 'dst_in_deg', 'dst_out_deg']


db_CIC0 = {
    'feats': {
        0:  # sigma
            {
                0: ['src_c_coeff_d2', 'dst_c_coeff_d1'],
                1: ['src_deg', 'dst_in_deg'],
                2: ['src_deg', 'dst_in_deg']
            },
        1000:
            {
                0: ['src_c_coeff_d1', 'dst_betweenness'],
                1: ['src_out_deg', 'src_closeness', 'src_c_coeff_d1', 'dst_betweenness'],
                2: ['src_c_coeff_d1', 'dst_betweenness']

            },
        5000:
            {
                0: ['src_c_coeff_d1', 'dst_betweenness'],
                1: ['src_out_deg', 'src_c_coeff_d1', 'dst_betweenness'],
                2: ['src_c_coeff_d1', 'dst_betweenness']

            },
    },
    'params': {
        0:  # sigma
            {
                0: {'C': 100, 'gamma': 1},
                1: {'C': 1, 'gamma': 0.1},
                2: {'C': 1, 'gamma': 0.1}
            },
        1000:
            {
                0: {'C': 100, 'gamma': 1},
                1: {'C': 1, 'gamma': 1},
                2: {'C': 100, 'gamma': 1},
            },
        5000:
            {
                0: {'C': 100, 'gamma': 1},
                1: {'C': 1, 'gamma': 1},
                2: {'C': 100, 'gamma': 1}
            }
    }
}
