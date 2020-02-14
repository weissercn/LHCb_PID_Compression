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

parser = argparse.ArgumentParser(description='LHCb PID compression script')
parser.add_argument('particle_type', choices=['pion', 'electron', 'muon', 'ghost', 'proton', 'kaon'],
                    help='particle type for dataset')
parser.add_argument('input', type=str,
                    help='File to compress')
parser.add_argument('output', type=str, help='Path to save')

args = parser.parse_args()

CRAMER_DIM = 150

# input parameters
particle_type = args.particle_type
input_path = args.input
output_path = args.output

# build-in parameters
pids = {'a':'pion', 'b': 'electron', 'c': 'muon', 'd': 'ghost', 'e': 'proton', 'f': 'kaon'}
inv_pids = {v:k for k,v in pids.items()}
pid = inv_pids[particle_type]
in_columns = ['S0aux7', 'S0aux6', 'S3aux0', 'S2aux0', 'S5aux0']
out_columns = ['0', '1', '2']
IN_DIM = len(in_columns)
OUT_DIM = len(out_columns)
LATENT_DIMENSIONS = 3

# load models
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

output = preprocessor.inverse_transform(np.concatenate([output_gen, data[preprocessor_columns].values[:, OUT_DIM:]], axis=1))[:,:OUT_DIM]
pd.DataFrame(output).to_csv(output_path, index=False, header=None)