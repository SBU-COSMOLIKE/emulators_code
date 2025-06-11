import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cobaya.theory import Theory
#from cobaya.theories.cosmo import BoltzmannBase
from cobaya.typing import InfoDict
from cobaya.theories.emulsn.emulator import Supact, Affine, Better_Attention, Better_Transformer, ResBlock, ResMLP, TRF
import joblib
import scipy
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict

class emulsn(Theory):   
    renames: Mapping[str, str] = empty_dict
    extra_args: InfoDict = { }
    _must_provide: dict
    path: str
    
    def initialize(self):
        super().initialize()
        self.ordering = self.extra_args.get('ordering')
        self.z_lin_dl = self.extra_args.get('zlindl')
        self.extradllevel = self.extra_args.get('extradllevel')

        self.ROOT = os.environ.get("ROOTDIR")
        self.PATH4 = self.ROOT + "/" + self.extra_args.get('dlfilename')
        self.extrainfo_dl = np.load(self.ROOT+"/"+self.extra_args.get('dlextraname'), allow_pickle=True)
        self.transmat_dl  = np.load(self.ROOT+"/"+self.extra_args.get('dltransmat'), allow_pickle=True)
     
        intdim = 4
        nlayer = 4

        device = 'cpu'

        self.model4 = ResMLP(input_dim=len(self.ordering), output_dim=len(self.transmat_dl), int_dim=intdim, N_layer=nlayer)
        self.model4 = self.model4.to(device)
        self.model4 = nn.DataParallel(self.model4)
        self.model4.load_state_dict(torch.load(self.PATH4+'.pt',map_location=device))
        self.model4 = self.model4.module.to(device)
        self.model4.eval()

        self.testh0 = -1

    def get_allow_agnostic(self):
        return True

    def get_requirements(self):
        return {
          "H0": None,
          "omegam": None
        }

    def predict_dl(self,model,X, extrainfo,transform_matrix):
        device = 'cpu'
        X_mean=torch.Tensor(extrainfo.item()['X_mean']).to(device)
        X_std=torch.Tensor(extrainfo.item()['X_std']).to(device)
        Y_mean=extrainfo.item()['Y_mean']
        Y_std=extrainfo.item()['Y_std']
        Y_mean_2=torch.Tensor(extrainfo.item()['Y_mean_2']).to(device)
        Y_std_2=torch.Tensor(extrainfo.item()['Y_std_2']).to(device)
        X = torch.Tensor(X).to(device)

        with torch.no_grad():
            X_norm=((X - X_mean) / X_std)
            X_norm.to(device)
            pred=model(X_norm)
            M_pred=pred.to(device)
            y_pred = (M_pred.float() *Y_std_2.float()+Y_mean_2.float()).cpu().numpy()
            y_pred = np.matmul(y_pred,transform_matrix)*Y_std+Y_mean
            y_pred = np.exp(y_pred)-self.extradllevel
        return y_pred[0]
  
    def calculate(self, state, want_derived=True, **params):
        cmb_param   = params.copy()
        cmb_params  = np.array([cmb_param[key] for key in self.ordering])
        state["dl"] = self.predict_dl(self.model4, cmb_params, self.extrainfo_dl, self.transmat_dl)
        return True
    
    def get_angular_diameter_distance(self,z):
        d_l = self.current_state["dl"].copy()
        z_lin = np.load(self.z_lin_dl, allow_pickle=True)
        d_a = d_l/(1+z_lin)**2
        D_A_interpolate = interpolate.interp1d(z_lin, d_a)
        return D_A_interpolate(z)
    