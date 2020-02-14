import tensorflow as tf
import os
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
parser.add_argument('particle_type', choices=['kaon', 'electron', 'muon'],
                    help='particle type for dataset')
parser.add_argument('input', type=str,
                    help='File to compress')
parser.add_argument('output', type=str, help='Path to save')

args = parser.parse_args()

# input parameters
input_path = args.input
output_path = args.output

# build-in parameters
pids = {'k':'kaon', 'e': 'electron', 'm': 'muon'}
inv_pids = {v:k for k,v in pids.items()}
pid = inv_pids[args.particle_type]
in_columns = ['S0aux7', 'S0aux6', 'S3aux0', 'S2aux0', 'S5aux0']
out_columns = ['0', '1', '2']

vars_list_input = ['GS3x1', 'GS3x0', 'GS0x7']
vars_list_aux = ['S5aux0', 'S3aux0', 'S2aux0', 'S0aux0', 'S0aux1', 'S0aux2', 'S0aux3', 'S2aux1', 'S2aux2', 'S2aux3', 'S0aux4', 'S0aux5', 'S0aux6', 'S0aux7', 'S0aux8']
vars_list_aux_gan = ['S0aux7', 'S0aux6', 'S3aux0', 'S2aux0', 'S5aux0']

# GAN parameters
IN_DIM = len(in_columns)
OUT_DIM = len(out_columns)
LATENT_DIMENSIONS = 3

# 1. GAN
MODEL_NAME = "GAN_Kramer_pid{}".format(inv_pids[args.particle_type])
MODEL_WEIGHTS_FILE = "./weights/%s.pkl" % MODEL_NAME
# load GAN models
def get_dense(num_layers):
    return [ll.Dense(80, activation='relu') for i in range(num_layers)]

with open(MODEL_WEIGHTS_FILE, 'rb') as f: weights = pickle.load(f)
generator = keras.models.Sequential(
    [ll.InputLayer([LATENT_DIMENSIONS + IN_DIM])] + get_dense(3) +
        [ll.Dense(OUT_DIM)])
generator.set_weights(weights)


# read data
data = pd.read_csv(input_path)

robust_scaler = joblib.load(os.path.join('gan_preprocessors', MODEL_NAME) + "_robust_preprocessor.pkl") 
max_abs_scaler = joblib.load(os.path.join('gan_preprocessors', MODEL_NAME) + "_maxabs_preprocessor.pkl")

for var in vars_list_aux+vars_list_input:
    if var not in data.columns:
        data[var] = [0]*len(data)

data_scaled = data[vars_list_aux+vars_list_input].copy()
data_scaled = pd.DataFrame(robust_scaler.transform(data_scaled), columns=data_scaled.columns)
data_scaled = pd.DataFrame(max_abs_scaler.transform(data_scaled), columns=data_scaled.columns)

input_data = data_scaled[in_columns]

input_noise = np.random.randn(len(input_data), LATENT_DIMENSIONS)
input_gen = np.concatenate([input_noise, input_data], axis=1)
output_gen = generator.predict(input_gen)

output_transformed = max_abs_scaler.inverse_transform(np.concatenate([data_scaled[vars_list_aux].values, output_gen], axis=1))
output_transformed = robust_scaler.inverse_transform(output_transformed)
gan_output = output_transformed[:, -OUT_DIM:]

    
pd.DataFrame(np.concatenate([gan_output, data[vars_list_aux]], axis=1), columns=vars_list_input+vars_list_aux).to_csv(args.output)