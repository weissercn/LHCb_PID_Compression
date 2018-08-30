from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import numpy as np
import pandas as pd
import tensorflow as tf

datasets = [
    "/mnt/nkazeev/Dirc_two_particles_v4/dirc_two_patricles_v3_2018-07-20T19:17:27.109954.hdf5",
    "/mnt/nkazeev/Dirc_two_particles_v4/dirc_two_patricles_v3_2018-07-20T21:16:11.660071.hdf5",
    "/mnt/nkazeev/Dirc_two_particles_v4/dirc_two_patricles_v3_2018-07-20T22:43:36.495871.hdf5",
    "/mnt/nkazeev/Dirc_two_particles_v4/dirc_two_patricles_v3_2018-07-21T01:23:58.138218.hdf5",
    "/mnt/nkazeev/Dirc_two_particles_v4/dirc_two_patricles_v3_2018-07-21T19:44:48.893343.hdf5",
    "/mnt/nkazeev/Dirc_two_particles_v4/dirc_two_patricles_v3_2018-07-22T12:21:10.862541.hdf5",
    "/mnt/nkazeev/Dirc_two_particles_v4/dirc_two_patricles_v3_2018-07-23T07:26:41.140862.hdf5",
    "/mnt/nkazeev/Dirc_two_particles_v4/dirc_two_patricles_v3_2018-07-23T21:24:49.932229.hdf5"
]

dll_columns = ['S0aux7', 'S0aux6', 'S3aux0', 'S2aux0', 'S5aux0']
raw_feature_columns = [
    'particle_one_type', 'particle_one_energy', 'particle_two_energy',
    'particle_one_eta', 'particle_two_eta', 'particle_one_x', 'particle_two_x']
non_type_features = [
    'particle_one_energy', 'particle_two_energy',
    'particle_one_eta', 'particle_two_eta', 'particle_one_x', 'particle_two_x']
                     
                     
y_count = len(dll_columns)
TEST_SIZE = 0.5
ENERGY_CUT = 2.5

def load_and_cut(file_name):
    data = pd.read_csv(file_name)
#     data = data[(data.particle_one_energy > ENERGY_CUT)]
#     assert not ((data[['support_electron', 'support_kaon', 'support_muon',
#                        'support_proton', 'support_pion']] == 0).any()).any()
    return data.drop("pid", axis=1)

def load_and_preprocess_dataset(file_name):
    data = load_and_cut(file_name)
#     data = pd.get_dummies(data, prefix=['input'],
#                           columns=], drop_first=True)
#     assert (data.columns == ['dll_electron', 'dll_kaon', 'dll_muon', 'dll_proton', 'dll_bt',
#                              'particle_one_energy', 'particle_two_energy', 'particle_one_eta',
#                              'particle_two_eta', 'particle_one_x', 'particle_two_x',
#                              'particle_one_type_1',
#                              'particle_one_type_2', 'particle_one_type_3', 
#                              'particle_one_type_4']).all()
    return data


def split(data):
    data_train, data_val = train_test_split(data, test_size=TEST_SIZE, random_state=42)
    data_val, data_test = train_test_split(data_val, test_size=TEST_SIZE, random_state=1812)
    return data_train, data_val, data_test

def split_and_scale(data):
    data_train, data_val = train_test_split(data, test_size=TEST_SIZE, random_state=42)
    scaler = RobustScaler().fit(data_train)
    # pandas...
    data_train = pd.DataFrame(scaler.transform(data_train.values),
                              columns=data_train.columns)
    data_val = pd.DataFrame(scaler.transform(data_val.values),
                            columns=data_val.columns)
    data_val, data_test = train_test_split(data_val, test_size=TEST_SIZE, random_state=1812)
    return data_train, data_val, data_test, scaler

def get_tf_dataset(dataset, batch_size):
    shuffler = tf.contrib.data.shuffle_and_repeat(dataset.shape[0])
    suffled_ds = shuffler(tf.data.Dataset.from_tensor_slices(dataset))
    return suffled_ds.batch(batch_size).prefetch(1).make_one_shot_iterator().get_next()
#     return tf.data.Dataset.from_tensor_slices(dataset).prefetch(1).make_one_shot_iterator().get_next()

def select_by_cuts(data, variable_cut):
    selected_data = np.ones(len(data), dtype=np.bool)
    for variable, cut in variable_cut.items():
        selected_data &= (data[variable] < cut[1]) & (data[variable] >= cut[0])
    return data.loc[selected_data]

def scale_pandas(dataframe, scaler):
    return pd.DataFrame(scaler.transform(dataframe.values), columns=dataframe.columns)

def get_scaled_typed_dataset(dataset_index, particle_type, dtype=None):
    """
    particle_one_type according to fastDirc:
    Electron = 0,
    Muon = 1,
    Pion = 2,
    Kaon = 3,
    Proton = 4
    """
    data_full = load_and_cut(datasets[dataset_index])
    # Must split the whole to preserve train/test split""
    data_train, data_val, _ = split(data_full)
    data_train = data_train[data_train.particle_one_type == particle_type].drop(
        columns="particle_one_type")
    data_val = data_val[data_val.particle_one_type == particle_type].drop(columns="particle_one_type")
    scaler = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=int(1e6),
                                 subsample=int(1e10)).fit(data_train.values)
    data_train = scale_pandas(data_train, scaler)
    data_val = scale_pandas(data_val, scaler)
    if dtype is not None:
        data_train = data_train.astype(dtype, copy=False)
        data_val = data_val.astype(dtype, copy=False)
    return data_train, data_val, scaler

def get_scaled_dataset(dataset_index):
    data_full = pd.read_hdf(datasets[dataset_index])
    data_full = data_full[(data_full.particle_one_energy > ENERGY_CUT)]
    p1_type_orig = np.copy(data_full.particle_one_type.values)
    p2_type_orig = np.copy(data_full.particle_two_type.values)
    # TOOD rewrite notebooks and use scaler on train/test
    scaler = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=int(1e6),
                                 subsample=int(1e10), copy=False)
    data_full_np = scaler.fit_transform(data_full)
    res = pd.DataFrame(data_full_np, columns=data_full.columns)
    res.particle_one_type = p1_type_orig
    res.particle_two_type = p2_type_orig
    return res