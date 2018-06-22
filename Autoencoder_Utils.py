#Useful functions live here. 

#Importing necessary libraries
import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, TensorBoard
from keras import optimizers
from keras import backend as K
import keras

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics

import math
import pandas as pd

import matplotlib.pyplot as plt
#%matplotlib notebook

import string
plt.rc('text', usetex=False)

import pickle




def print_features_histograms(features,target=None, save_filename=None, normed=True):
    hist_params = {'normed': normed, 'bins': 60, 'alpha': 0.4}
    # create the figure
    fig = plt.figure(figsize=(8,  2*math.ceil(features.shape[1] / 2.)))
    
    
    for n, feature in enumerate(features):
        # add sub plot on our figure
        ax = fig.add_subplot(math.ceil(features.shape[1] / 2.), 2, n + 1)
        # define range for histograms by cutting 1% of data from both ends

        min_value, max_value = np.percentile(features[feature], [1, 99])
        if target is not None:
            min_value2, max_value2 = np.percentile(target[feature], [1, 99])
            min_value, max_value = min(min_value, min_value2), max(max_value, max_value2)
        min_value-=0.1*np.abs(max_value)
        max_value+=0.2*np.abs(max_value)
        ax.hist(features[feature], range=(min_value, max_value), 
                 label='predicted', **hist_params)
        if target is not None:
            ax.hist(target[feature], range=(min_value, max_value), 
                 label='target', **hist_params)
        ax.set_title(feature)
    plt.subplots_adjust(top=0.80, bottom=0.08, left=0.10, right=0.95, hspace=0.60, wspace=0.35)
    
    if save_filename is not None: plt.savefig(save_filename)
        
def print_features_histograms_displ(features,target=None, save_filename=None, normed=True):
    n_in_row = 3
    n_in_col = 3
    
    n_features_left = features.shape[1]
    n_features_used = 0
    n_turns = 0
    
    while n_features_left>0:
        
        n_features_used_now = min(n_in_row*n_in_col, n_features_left)
        hist_params = {'normed': normed, 'bins': 60, 'alpha': 0.4}
        # create the figure
        fig = plt.figure(figsize=(8,  n_in_row*math.ceil(n_features_used_now *1./ n_in_row)))

        for n in range(n_features_used, n_features_used+n_features_used_now):
            feature = features.keys()[n]
            #for n, feature in enumerate(features):
            # add sub plot on our figure
            ax = fig.add_subplot(math.ceil(n_features_used_now *1./ n_in_row), n_in_row, n+1-n_features_used )
            # define range for histograms by cutting 1% of data from both ends
            min_value, max_value = np.percentile(features[feature], [1, 99])
            if target is not None:
                min_value2, max_value2 = np.percentile(target[feature], [1, 99])
                min_value, max_value = min(min_value, min_value2), max(max_value, max_value2)
            min_value-=0.1*np.abs(max_value)
            max_value+=0.2*np.abs(max_value)
            ax.hist(features[feature], range=(min_value, max_value), 
                     label='predicted', **hist_params)
            if target is not None:
                ax.hist(target[feature], range=(min_value, max_value), 
                     label='target', **hist_params)
            ax.set_title(feature)
        plt.subplots_adjust(top=0.80, bottom=0.08, left=0.10, right=0.95, hspace=0.60, wspace=0.35)

        if save_filename is not None: plt.savefig(str(n_turns)+save_filename)
        n_features_left -= n_features_used_now
        n_features_used += n_features_used_now
        n_turns +=1

def plot_difference_displ(TYPE, decoded, orig, encoding_dim, TYPE_FEATURES="ALL", FTS_SCLD=False, SetMinMax=False, Transform=True, l_minmax=[[-15,15],[-80,110],[-15,15],[-80,80],[-0.8,1],[-.8,.8],[-.8,.8],[-.8,.8],[-.8,.8]]):
    from matplotlib.colors import LogNorm
    
    
    #decoded,orig
    if Transform:
        unscaled_decoded = fs.invtransform(decoded.values)
        unscaled_orig    = fs.invtransform(orig.values)
        features = pd.DataFrame(unscaled_decoded, columns=orig.columns) 
        target = pd.DataFrame(unscaled_orig, columns=orig.columns) 
    else:
        features = pd.DataFrame(decoded.values, columns=orig.columns) 
        target = pd.DataFrame(orig.values, columns=orig.columns) 
        
    n_in_row = 3
    n_in_col = 3
    
    n_features_left = features.shape[1]
    n_features_used = 0
    n_turns = 0
        
    #print orig.columns
    #hist_params = {'normed': True, 'bins': 60, 'alpha': 0.4}
    hist_params = {'bins': 60, 'alpha': 0.4}

    while n_features_left>0:
        
        n_features_used_now = min(n_in_row*n_in_col, n_features_left)
        # create the figure
        fig = plt.figure(figsize=(9,  n_in_row*math.ceil(n_features_used_now *1./ n_in_row)-1))

        n_points = len(features.index)
        #print n_points


        for n in range(n_features_used, n_features_used+n_features_used_now):
        #for n, feature in enumerate(features):
            feature = features.keys()[n]
            # add sub plot on our figure

            ax = fig.add_subplot(math.ceil(n_features_used_now *1./ n_in_row), n_in_row, n+1-n_features_used )
            #ax = fig.add_subplot(math.ceil(features.shape[1] / 3.), 3, n + 1)
            # define range for histograms by cutting 1% of data from both ends
            min_value, max_value = np.percentile(features[feature], [1, 99])
            if target is not None:
                min_value2, max_value2 = np.percentile(target[feature], [1, 99])
                min_value, max_value = min(min_value, min_value2), max(max_value, max_value2)
            min_value-=0.1*max_value
            max_value+=0.2*max_value
            
            
            

            if False:
                ax.hist(features[feature], range=(min_value, max_value), 
                         label='predicted', **hist_params)
                if target is not None:
                    ax.hist(target[feature], range=(min_value, max_value), 
                         label='target', **hist_params)
            #print target[feature]
            #print np.abs(target[feature] - features[feature])
            #h = ax.hist2d(target[feature], target[feature] - features[feature], bins=[15,15], norm=LogNorm(vmin=1, vmax=n_points) )
            h = ax.hist2d(target[feature], target[feature] - features[feature], bins=[15,15], norm=LogNorm(vmin=1, vmax=n_points),cmap='inferno'  )
            if SetMinMax: ax.set_ylim(l_minmax[n][0],l_minmax[n][1])
            ax.set_xlabel(feature)
            ax.set_ylabel('Delta')
            if FTS_SCLD: ax.set_xlim(-0.75,0.85)

            #ax.set_title(feature)

        #plt.subplots_adjust(top=0.89, bottom=0.08, left=0.10, right=0.9, hspace=0.60, wspace=0.35)
        #cbar_ax = fig.add_axes([0.1, 0.95, 0.8, 0.03])
        #cbar = plt.colorbar(h[3], cax=cbar_ax, orientation='horizontal', ticks=[1, 2, 10, 20, 100, 270])
        #cbar.ax.set_xticklabels([1, 2, 10, 20, 100, 270])# horizontal colorbar

        plt.subplots_adjust(top=0.95, bottom=0.10, left=0.05, right=0.83, hspace=0.3, wspace=0.2)
        cbar_ax = fig.add_axes([0.88, 0.05, 0.05, 0.9])
        cbar = plt.colorbar(h[3], cax=cbar_ax, ticks=[1, 10, 100, 1000, 10000, 100000, 200000])
        cbar.ax.set_yticklabels(['1', '10', '100', '1k', '10k', '100k', '200k'])# horizontal colorbar

        plt.savefig(str(n_turns)+"FeatureDeviation_{}_{}.png".format(encoding_dim, TYPE))
        
        
        n_features_left -= n_features_used_now
        n_features_used += n_features_used_now
        n_turns +=1
        
def roc_curves_old(TYPE, decoded, orig, truth, encoding_dim, FIX_POS_Mu=True):
    
    #from collections import Counter
    #print Counter(truth)
    
    #decoded,orig
    
    unscaled_decoded = fs.invtransform(decoded.values)
    unscaled_orig    = fs.invtransform(orig.values)
    
    features = pd.DataFrame(unscaled_decoded, columns=df.columns[1:]) 
    target = pd.DataFrame(unscaled_orig, columns=df.columns[1:]) 
    
    #print orig.columns
    hist_params = {'normed': True, 'bins': 60, 'alpha': 0.4}
    # create the figure
    fig = plt.figure(figsize=(14*2./3.,  2.*math.ceil(features.shape[1] / 3.)))
    
    
    for n, feature in enumerate(features):
        # add sub plot on our figure
        ax = fig.add_subplot(math.ceil(features.shape[1] / 3.), 3, n + 1)
        # define range for histograms by cutting 1% of data from both ends
        #print features[feature]
        #print target[feature]
        
        if FIX_POS_Mu: apos_label = 13
        else:          apos_label = fig_to_corr_pid[n]
        
        fpr_dec, tpr_dec, thresholds_dec = metrics.roc_curve(truth, features[feature], pos_label=apos_label)
        fpr_orig, tpr_orig, thresholds_orig = metrics.roc_curve(truth, target[feature], pos_label=apos_label)
        
        
        roc_auc_dec = metrics.auc(fpr_dec, tpr_dec)
        roc_auc_orig = metrics.auc(fpr_orig, tpr_orig)
        
        ax.plot(fpr_dec , tpr_dec,  "--", color='blue', label='dec %0.2f'  % roc_auc_dec)
        ax.plot(fpr_orig, tpr_orig, "--", color='red',  label='orig %0.2f' % roc_auc_orig)
        
        ax.legend()
        
        
        
        ax.set_xlabel(feature +" FPR")
        ax.set_ylabel('TPR')
        
        #ax.set_title(feature)
    plt.subplots_adjust(top=0.95, bottom=0.10, left=0.05, right=0.975, hspace=0.3, wspace=0.2)
    
    plt.savefig("ROCs_{}_{}.png".format(encoding_dim, TYPE))
    

def create_autoencoder_aux(n_features, encoding_dim, n_aux_features=5, p_drop=0.5, n_layers=3, thickness=2  ):
    # build encoding model using keras where we can feed in auxilliary info
    inputs = Input(shape=(n_features,), name='main_input')
    
    # "encoded" is the encoded representation of the input
    x = inputs
    """x = Dense(2*n_features, activation='tanh')(inputs)
    x = Dense(2*encoding_dim, activation='tanh')(x)
    x = Dropout(p_drop)(x)"""
    #encoded = Dense(encoding_dim, activation='tanh')(x)
    
    #https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
    aux_inputs = Input(shape=(n_aux_features,), name='aux_inputs')
    x = keras.layers.concatenate([x, aux_inputs])
    

    for i in range(n_layers-1):
        x = Dense(thickness*n_features, activation='tanh')(x)
        x = keras.layers.concatenate([x, aux_inputs])

    #x = Dropout(p_drop)(x)
    
    x = Dense(thickness*encoding_dim, activation='tanh')(x)
    x = keras.layers.concatenate([x, aux_inputs])
    #x = Dropout(p_drop)(x)
    encoded = Dense(encoding_dim, activation='tanh', name='encoded')(x)

    # "decoded" is the lossy reconstruction of the input
    x = encoded
    """x = Dense(2*encoding_dim, activation='tanh')(encoded)
    x = Dropout(p_drop)(x)
    x = Dense(2*n_features, activation='tanh')(x)"""
    #decoded = Dense(n_features, activation='tanh')(x)
    
    x = keras.layers.concatenate([x, aux_inputs])
    
    x = Dense(thickness*encoding_dim, activation='tanh')(x)
    #x = Dropout(p_drop)(x)
    
    for i in range(n_layers-1):
        x = keras.layers.concatenate([x, aux_inputs])
        x = Dense(thickness*n_features, activation='tanh')(x)
        

    #x = Dropout(p_drop)(x)
    decoded = Dense(n_features, activation='tanh')(x)
    

    # this model maps an input to its reconstruction
    autoencoder = Model([inputs, aux_inputs ], decoded)

    # this model maps an input to its encoded representation
    encoder = Model([inputs, aux_inputs], encoded)
    
    
    if False:
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_n_layers = len(autoencoder.layers) - len(encoder.layers)
        #print "decoder_n_layers : ", decoder_n_layers
        decoder_layers = autoencoder.layers[-decoder_n_layers:]
        decoding = encoded_input
        for i in decoder_layers:
            decoding= i(decoding)


        # create the decoder model
        decoder = Model([encoded_input, aux_inputs], decoding)
        
    else:
        decoder = K.function([encoded, aux_inputs, K.learning_phase()], [decoded])


    optimizer_adam = optimizers.Adam(lr=0.001)

    autoencoder.compile(loss='mse', optimizer=optimizer_adam)
    
    #autoencoder.compile(optimizer=optimizer_adam, 
    #        loss={'decoded': 'mse'},
    #        loss_weights={'decoded': 1.})
    
    return autoencoder, encoder, decoder

def create_autoencoder_aux_skipPNN(n_features, encoding_dim, n_aux_features=5, n_pnn_features=5,  p_drop=0.5, n_layers=3, thickness=2  ):
    # build encoding model using keras where we can feed in auxilliary info
    inputs = Input(shape=(n_features,), name='main_input')
    
    # "encoded" is the encoded representation of the input
    x = inputs
    """x = Dense(2*n_features, activation='tanh')(inputs)
    x = Dense(2*encoding_dim, activation='tanh')(x)
    x = Dropout(p_drop)(x)"""
    #encoded = Dense(encoding_dim, activation='tanh')(x)
    
    #https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
    aux_inputs = Input(shape=(n_aux_features,), name='aux_inputs')
    x = keras.layers.concatenate([x, aux_inputs])
    

    for i in range(n_layers-1):
        x = Dense(thickness*n_features, activation='tanh')(x)
        x = keras.layers.concatenate([x, aux_inputs])

    #x = Dropout(p_drop)(x)
    
    x = Dense(thickness*encoding_dim, activation='tanh')(x)
    x = keras.layers.concatenate([x, aux_inputs])
    #x = Dropout(p_drop)(x)
    encoded = Dense(encoding_dim, activation='tanh', name='encoded')(x)

    # "decoded" is the lossy reconstruction of the input
    x = encoded
    """x = Dense(2*encoding_dim, activation='tanh')(encoded)
    x = Dropout(p_drop)(x)
    x = Dense(2*n_features, activation='tanh')(x)"""
    #decoded = Dense(n_features, activation='tanh')(x)
    
    x = keras.layers.concatenate([x, aux_inputs])
    
    x = Dense(thickness*encoding_dim, activation='tanh')(x)
    #x = Dropout(p_drop)(x)
    
    for i in range(n_layers-1):
        x = keras.layers.concatenate([x, aux_inputs])
        x = Dense(thickness*(n_features+n_pnn_features), activation='tanh')(x)
        

    #x = Dropout(p_drop)(x)
    decoded = Dense(n_features, activation='tanh', name='decoded')(x)
    
    pnn_decoded = Dense(n_pnn_features, activation='tanh', name='pnn_decoded')(x)
    

    # this model maps an input to its reconstruction
    autoencoder = Model([inputs, aux_inputs ], [decoded, pnn_decoded] )

    # this model maps an input to its encoded representation
    encoder = Model([inputs, aux_inputs], encoded)
    
    
    if False:
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_n_layers = len(autoencoder.layers) - len(encoder.layers)
        #print "decoder_n_layers : ", decoder_n_layers
        decoder_layers = autoencoder.layers[-decoder_n_layers:]
        decoding = encoded_input
        for i in decoder_layers:
            decoding= i(decoding)


        # create the decoder model
        decoder = Model([encoded_input, aux_inputs], decoding)
        
    else:
        decoder = K.function([encoded, aux_inputs, K.learning_phase()], [decoded, pnn_decoded])


    optimizer_adam = optimizers.Adam(lr=0.001)

    #autoencoder.compile(loss='mse', optimizer=optimizer_adam)
    
    autoencoder.compile(optimizer=optimizer_adam, 
            loss={'decoded': 'mse', 'pnn_decoded' : 'mse'},
            loss_weights={'decoded': 1., 'pnn_decoded': 1.})
    
    return autoencoder, encoder, decoder


