"""
This file  generate_FLOW_gencut_ksttrain_nspd.py is like generate_gencut_ksttrain_nspd.py but uses a flow-based generative model
Run like:
python3 generate_gencut_new.py kaon /data/weisser/LbVMWeisser_shared/Analyses/LUV_ML/NTuple_BKee_481_482_VAE_K_out.csv test.csv pidk
"""

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

import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedUMNNAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn.nets import ResidualNet


parser = argparse.ArgumentParser(description='LHCb PID compression script')
parser.add_argument('particle_type', choices=['kaon', 'electron', 'muon'],
                    help='particle type for dataset')
parser.add_argument('input', type=str,
                    help='File to compress')
parser.add_argument('output', type=str, help='Path to save')
parser.add_argument('cut_type', choices=['none', 'pidmu', 'pide', 'pnnk'],
                    help='cut on VAE output after generation')
args = parser.parse_args()

# input parameters
input_path = args.input
output_path = args.output
cut_type = args.cut_type

# build-in parameters
pids = {'k':'kaon', 'e': 'electron', 'm': 'muon'}
inv_pids = {v:k for k,v in pids.items()}
pid = inv_pids[args.particle_type]
in_columns = ["S6aux0",'S0aux7', 'S0aux6', 'S3aux0', 'S2aux0', 'S5aux0']
out_columns = ['0', '1', '2']

vars_list_input = ['GS3x1', 'GS3x0', 'GS0x7']
vars_list_aux = ["S6aux0", 'S5aux0', 'S3aux0', 'S2aux0', 'S0aux0', 'S0aux1', 'S0aux2', 'S0aux3', 'S2aux1', 'S2aux2', 'S2aux3', 'S0aux4', 'S0aux5', 'S0aux6', 'S0aux7', 'S0aux8']
vars_list_aux_gan = ["S6aux0", 'S0aux7', 'S0aux6', 'S3aux0', 'S2aux0', 'S5aux0']

# GAN parameters
IN_DIM = len(in_columns)
OUT_DIM = len(out_columns)
LATENT_DIMENSIONS = 3

# 1. GAN
MODEL_NAME = "Flow_Kramer_pid{}_ksttrain_nspd".format(inv_pids[args.particle_type])
MODEL_WEIGHTS_FILE = "./weights/%s.pt" % MODEL_NAME

flowbest = torch.load(MODEL_WEIGHTS_FILE)
flowbest.eval()


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


###########################################
## TESTING WHAT THE CUTOFFS ARE AFTER SCALING
#print("type(data[vars_list_aux+vars_list_input]) ", type(data[vars_list_aux+vars_list_input]))
data_scaled_test = data[vars_list_aux+vars_list_input].iloc[[0,1]].copy()
data_scaled_test['GS3x1'][0] = -3
data_scaled_test['GS3x0'][0] = 0
data_scaled_test['GS0x7'][0] = 0
data_scaled_test['GS0x7'][1] = 1
#print("data_scaled_test ", data_scaled_test.shape, "\n",data_scaled_test)
data_scaled_test = pd.DataFrame(robust_scaler.transform(data_scaled_test), columns=data_scaled_test.columns)
data_scaled_test = pd.DataFrame(max_abs_scaler.transform(data_scaled_test), columns=data_scaled_test.columns)
#print("data_scaled_test ", data_scaled_test.shape, "\n",data_scaled_test)
GS3x1_m3 = data_scaled_test['GS3x1'][0]
GS3x0_0 = data_scaled_test['GS3x0'][0]
GS0x7_0 = data_scaled_test['GS0x7'][0]
GS0x7_1 = data_scaled_test['GS0x7'][1]
#print("GS3x1_m3 : ", GS3x1_m3)

#GS3x1 = -3 turns into GS3x1 = -0.14385 after scaling
###########################

input_data_orig = data_scaled[in_columns]

n_data, n_cols = input_data_orig.shape 
passcut_mask = np.array([False]*n_data) # This is the mask of whether events pass the cut

output_gen = np.full((n_data, 3), None)  # pd.DataFrame({'GS3x1':[None]*n_data, 'GS3x0':[None]*n_data, 'GS0x7': [None]*n_data } )
i_gen = 0

while np.sum(np.logical_not(passcut_mask))>0:
    # throw random numbers and generate again for events that don't pass the cut
    input_data = input_data_orig[np.logical_not(passcut_mask)] 
    #input_noise = np.random.randn(len(input_data), LATENT_DIMENSIONS)
    #input_gen = np.concatenate([input_noise, input_data], axis=1)
    #output_gen_part = generator.predict(input_gen)
    
    output_gen_part = flowbest.sample(1, context=torch.tensor(input_data.to_numpy(),dtype=torch.float32)).numpy().reshape((input_data.shape[0],-1))


    
    if cut_type== 'pidmu' :
        # vars_list_input : ['GS3x1', 'GS3x0', 'GS0x7'] =  ["trks_EcalPIDmu",  "trks_EcalPIDe",  "trks_ProbNNk"]
        tmp_passcut_mask = GS3x1_m3 < output_gen_part[:,0] # -3 before scaling  # trks_EcalPIDmu > -3   # length is number of newly generated data
    elif cut_type== 'pide':
        tmp_passcut_mask = GS3x0_0 < output_gen_part[:,1] 
    elif cut_type== 'pnnk':
        tmp_passcut_mask = np.logical_and((GS0x7_0 < output_gen_part[:,2]), (output_gen_part[:,2] < GS0x7_1))
    else:
        tmp_passcut_mask = np.array([True]*n_data) # length is number of newly generated data


    new_passcut_mask = np.array([False]*n_data) # length is total number of examples
    new_passcut_mask[np.logical_not(passcut_mask)] = tmp_passcut_mask
 
    output_gen[new_passcut_mask] = output_gen_part[tmp_passcut_mask]
    passcut_mask = np.logical_or(passcut_mask, new_passcut_mask)
    i_gen +=1
    print("i_gen ", i_gen, passcut_mask.shape, np.sum(passcut_mask))

"""
input_noise = np.random.randn(len(input_data), LATENT_DIMENSIONS)
input_gen = np.concatenate([input_noise, input_data], axis=1)
output_gen = generator.predict(input_gen)
"""

output_transformed = max_abs_scaler.inverse_transform(np.concatenate([data_scaled[vars_list_aux].values, output_gen], axis=1))
output_transformed = robust_scaler.inverse_transform(output_transformed)
gan_output = output_transformed[:, -OUT_DIM:]

    
pd.DataFrame(np.concatenate([gan_output, data[vars_list_aux]], axis=1), columns=vars_list_input+vars_list_aux).to_csv(args.output)
