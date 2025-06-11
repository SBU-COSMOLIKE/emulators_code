import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cobaya.theory import Theory
from cobaya.typing import InfoDict
from cobaya.theories.emulcmb.emulator import Supact, Affine, Better_Attention, Better_Transformer, ResBlock, ResMLP, TRF, CNNMLP
import joblib
import scipy
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict

class emultheta(Theory):
    renames: Mapping[str, str] = empty_dict
    extra_args: InfoDict = { }
    _must_provide: dict
    path: str
    
    def initialize(self):
        super().initialize()
        RT = os.environ.get("ROOTDIR")
        self.M      = [None]
        self.info   = [None]
        self.ord    = [None]
        self.req    = [] 

        fname  = RT + "/" + self.extra_args.get("file")[0]
        fextra = RT + "/" + self.extra_args.get("extra")[0]
        self.info[0] = np.load(fextra,allow_pickle=True)
        self.M[0]    = joblib.load(fname)
        self.ord[0]  = self.extra_args.get('ord')[0]
        self.req.extend(self.ord[0])

        self.req = list(set(self.req))
        d = {'omegamh2' : None}
        for i in self.req:
            d[i] = None
        self.req = d

    def get_requirements(self):
        return self.req

    def calculate(self, state, want_derived=False, **params):
        par = params.copy()

        params = self.ord[0]
        X_mean = self.info[0].item()['X_mean']
        Y_mean = self.info[0].item()['Y_mean']
        X_std  = self.info[0].item()['X_std']
        Y_std  = self.info[0].item()['Y_std']
        p =  np.array([par[key] for key in params]) - X_mean     
        H0 = self.M[0].predict(p/X_std)[0]*Y_std[0] + Y_mean[0]
        
        h2       = (H0/100.0)**2
        omegamh2 = par["omegamh2"]

        par.update({"H0": H0})
        par.update({"omegam": omegamh2/h2})   
        state.update({"H0": par["H0"]})
        state.update({"omegam": par["omegam"]})    
        state["derived"].update({"H0": par["H0"]})
        state["derived"].update({"omegam": par["omegam"]})   
        return True

    def get_H0(self):
        state = self.current_state.copy()
        return state["H0"]

    def get_omegam(self):
        state = self.current_state.copy()
        return state["omegam"]
 