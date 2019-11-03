import tensorflow as tf
import os
# os.environ['KERAS_BACKEND']='tensorflow'
import keras
import keras.layers as ll
import pandas as pd
import json
import pickle
import joblib
import numpy as np
import argparse

import keras
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

parser = argparse.ArgumentParser(description='LHCb PID compression script')
parser.add_argument('particle_type', choices=['pion', 'electron', 'muon', 'ghost', 'proton', 'kaon'],
                    help='particle type for dataset')
parser.add_argument('input', type=str,
                    help='File to compress')
parser.add_argument('output', type=str, help='Path to save')

args = parser.parse_args()

# input parameters
input_path = args.input
output_path = args.output

# build-in parameters
pids = {'a':'pion', 'b': 'electron', 'c': 'muon', 'd': 'ghost', 'e': 'proton', 'f': 'kaon'}
inv_pids = {v:k for k,v in pids.items()}
pid = inv_pids[args.particle_type]
in_columns = ['S0aux7', 'S0aux6', 'S3aux0', 'S2aux0', 'S5aux0']
out_columns = ['0', '1', '2']

# AE parameters
DATA_DIM = 35
ENCODING_DIM = 3
AUX_DIM = 15
N_LAYERS = 4
THICKNESS = 6

# GAN parameters
IN_DIM = len(in_columns)
OUT_DIM = len(out_columns)
LATENT_DIMENSIONS = 3

# 1. GAN

# load GAN models
def get_dense(num_layers):
    return [ll.Dense(80, activation='relu') for i in range(num_layers)]

with open('weights.pkl', 'rb') as f: weights = pickle.load(f)
generator = keras.models.Sequential(
    [ll.InputLayer([LATENT_DIMENSIONS + IN_DIM])] + get_dense(3) +
        [ll.Dense(OUT_DIM)])
generator.set_weights(weights[pid])
preprocessor = joblib.load(f'preprocessors/GAN_Kramer_dim3_bs1e4_n80-150_old_pid{pid}_preprocessor.pkl')


# read data
data = pd.read_csv(input_path)
with open('features_encoding.json') as f:
    features_encoding = json.load(f)
data.rename(columns=features_encoding, inplace=True)
    
# preprocess data
preprocessor_columns = '0	1	2	S5aux0	S3aux0	S2aux0	S0aux0	S0aux1	S0aux2	S0aux3	S2aux1	S2aux2	S2aux3	S0aux4	S0aux5	S0aux6	S0aux7	S0aux8'.split('\t')
replace_col = 'S0aux4'
target_columns = ['0', '1', '2']
assert(not replace_col in in_columns, 'Use another column to replace target columns')
preprocessor_columns = [col if col not in target_columns else replace_col for col in preprocessor_columns]
input_data = pd.DataFrame(preprocessor.transform(data[preprocessor_columns].values), columns=preprocessor_columns)[in_columns]

input_noise = np.random.randn(input_data.shape[0], LATENT_DIMENSIONS)
input_gen = np.concatenate([input_noise, input_data], axis=1)
output_gen = generator.predict(input_gen)

gan_output = preprocessor.inverse_transform(np.concatenate([output_gen, data[preprocessor_columns].values[:, OUT_DIM:]], axis=1))

# 2. AE

def create_autoencoder_aux(n_features, encoding_dim, n_aux_features=5, p_drop=0.5, n_layers=3, thickness=2):
    # encoder
    inputs = Input(shape=(n_features, ), name='main_input')
    aux_inputs = Input(shape=(n_aux_features, ), name='aux_inputs')
    
    
    x = keras.layers.concatenate([inputs, aux_inputs])

    for i in range(n_layers - 1):
        x = Dense(thickness * n_features, activation='tanh')(x)
        x = keras.layers.concatenate([x, aux_inputs])

    
    x = Dense(thickness * encoding_dim, activation='tanh')(x)
    x = keras.layers.concatenate([x, aux_inputs])

    encoded = Dense(encoding_dim, activation='tanh', name='encoded')(x)

    
    # decoder
    input_encoded = Input(shape=(encoding_dim, ))
    
    x = keras.layers.concatenate([input_encoded, aux_inputs])
    x = Dense(thickness * encoding_dim, activation='tanh')(x)
    
    for i in range(n_layers - 1):
        x = keras.layers.concatenate([x, aux_inputs])
        x = Dense(thickness * n_features, activation='tanh')(x)

    decoded = Dense(n_features, activation='tanh')(x)
    
    
    # models
    encoder = Model([inputs, aux_inputs], encoded, name="encoder")
    decoder = Model([input_encoded, aux_inputs], decoded, name="decoder")
    autoencoder = Model(
        [inputs, aux_inputs], decoder([encoder([inputs, aux_inputs]), aux_inputs]), name="autoencoder"
    )

    optimizer_adam = optimizers.Adam(lr=0.001)
    autoencoder.compile(loss='mse', optimizer=optimizer_adam)
    
    return autoencoder, encoder, decoder

autoencoder, encoder, decoder = create_autoencoder_aux(
    DATA_DIM, ENCODING_DIM, 
    n_aux_features=AUX_DIM, n_layers=N_LAYERS, thickness=THICKNESS
)

autoencoder.load_weights("model_old_ratio10_{}.hdf5".format(ENCODING_DIM))

print(gan_output.shape)
decoded_test = decoder.predict([gan_output[:, :LATENT_DIMENSIONS], gan_output[:, LATENT_DIMENSIONS:]])

# unscale output
vars_list_input = ['S0x0','S0x1','S0x2','S0x3','S0x4','S3x0','S3x1','S2x0','S2x1','S2x2','S2x3','S0x5','S0x6',
 'S0x7','S0x8','S0x9','S0x10','S1x0','S1x1','S1x2','S1x3','S1x4','S1x5','S5x0','S4x0','S4x1','S4x2','S3x2','S4x3',
 'S4x4','S5x1','S5x2','S5x3','S5x4','S4x5']

scalers = {var: {} for var in vars_list_input}
for i, var in enumerate(vars_list_input):
    scalers[var]['std'] = joblib.load(os.path.join('./preprocessors', "scaler_std_"+var) + ".pkl")
    scalers[var]['max'] = joblib.load(os.path.join('./preprocessors', "scaler_max_"+var) + ".pkl")
    
decoded_unscaled = decoded_test.copy()
for i, var in enumerate(vars_list_input):
    decoded_unscaled[:, i] = scalers[var]['std'].inverse_transform(
        scalers[var]['max'].inverse_transform(decoded_unscaled[:, i].reshape(-1, 1))
    ).reshape(-1)
    
pd.DataFrame(np.concatenate([decoded_unscaled, gan_output[:, LATENT_DIMENSIONS:]], axis=1), columns=vars_list_input+preprocessor_columns[LATENT_DIMENSIONS:]).to_csv(args.output)