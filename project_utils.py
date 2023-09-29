import pandas as pd
from os import mkdir


def make_directory(path: str):
    try:
        mkdir(path)
    except:
        pass


def load_and_etl_CIC_IDS_2017(user_path_CIC: str):
    """
    CIC-IDS2017 dataset load and pre-processing
    :param user_path_CIC: path of dataset files
    :return: pre-processed dataset in pandas.DataFrame format
    """
    friday1 = pd.read_csv(user_path_CIC + '/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                          encoding='cp1252', low_memory=False)
    friday2 = pd.read_csv(user_path_CIC + '/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                          encoding='cp1252', low_memory=False)
    friday3 = pd.read_csv(user_path_CIC + '/Friday-WorkingHours-Morning.pcap_ISCX.csv',
                          encoding='cp1252', low_memory=False)

    friday = pd.concat([friday1, friday2, friday3])

    monday = pd.read_csv(user_path_CIC + '/Monday-WorkingHours.pcap_ISCX.csv',
                         encoding='cp1252', low_memory=False)

    thursday1 = pd.read_csv(user_path_CIC + '/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                            encoding='cp1252', low_memory=False)
    thursday2 = pd.read_csv(user_path_CIC + '/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
                            encoding='cp1252', low_memory=False)

    thursday = pd.concat([thursday1, thursday2])

    tuesday = pd.read_csv(user_path_CIC + '/Tuesday-WorkingHours.pcap_ISCX.csv',
                          encoding='cp1252', low_memory=False)
    wednesday = pd.read_csv(user_path_CIC + '/Wednesday-workingHours.pcap_ISCX.csv',
                            encoding='cp1252', low_memory=False)

    processed_days = []
    for day in [monday, tuesday, wednesday, thursday, friday]:
        day.columns = day.columns.str.lstrip()
        day.columns = day.columns.str.replace(' ', '_')
        day.dropna(axis=0, how='all', inplace=True)
        day.fillna((day['Flow_Bytes/s'].mean()), inplace=True)
        day['Binary_Label'] = day['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

        day['Timestamp'] = pd.to_datetime(day['Timestamp'])
        day.sort_values(by=['Timestamp'], inplace=True)
        processed_days.append(day)
    return processed_days




