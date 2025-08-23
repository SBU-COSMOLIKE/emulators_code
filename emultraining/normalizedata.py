import numpy as np
from sklearn.decomposition import IncrementalPCA
import torch
camb_ell_min          = 2#30
camb_ell_max          = 3000
camb_ell_range        = camb_ell_max  - camb_ell_min 
#This script is for CMB TT TE and EE seperately.
train_samples=np.load('training input',allow_pickle=True)
train_data_vectors = np.load('training output',allow_pickle=True)

As_pos = # index of log A_s in parameter vector
tau_pos = # index of tau in parameter vector

A_s = np.exp(train_samples[:,As_pos])
exp2tau = np.exp(2*train_samples[:,tau_pos])


train_data_vectors = train_data_vectors/A_s[:,None]*exp2tau[:,None]

X_mean=train_samples.mean(axis=0, keepdims=True)
X_std=train_samples.std(axis=0, keepdims=True)
Y_mean=train_data_vectors.mean(axis=0, keepdims=True)
Y_std=train_data_vectors.std(axis=0, keepdims=True)

extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean}
np.save('Normalization dir',extrainfo)