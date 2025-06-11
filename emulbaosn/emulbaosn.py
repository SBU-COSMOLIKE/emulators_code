import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
#from cobaya.theories.cosmo import BoltzmannBase
from cobaya.theory import Theory
from cobaya.theories.emulbaosn.emulator import Supact, Affine, Better_Attention, Better_Transformer, ResBlock, ResMLP, TRF
import joblib
import scipy
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict

class emulbaosn(Theory):
    renames: Mapping[str, str] = empty_dict
    extra_args: InfoDict = {}
    _must_provide: dict
    path: str
    
    def initialize(self):
        super().initialize()
        RT = os.environ.get("ROOTDIR")
        self.M      = [None, None] # dl, H(z)
        self.info   = [None, None] # dl, H(z)
        self.z      = [None, None] # dl, H(z)
        self.tmat   = [None, None] # dl, H(z)
        self.offset = [None, None] # dl, H(z)
        self.ord    = [None, None] # dl, H(z)
        self.req    = [] 
        for i in range(2):
            if self.extra_args.get('eval')[i]:
                fname  = RT + "/" + self.extra_args.get("file")[i]
                fextra = RT + "/" + self.extra_args.get("extra")[i]
                fzlin  = RT + "/" + self.extra_args.get("zlin")[i]
                ftmat  = RT + "/" + self.extra_args.get("tmat")[i]
                
                self.info[i] = np.load(fextra, allow_pickle=True)
                self.z[i]    = np.load(fzlin, allow_pickle=True)
                self.tmat[i] = np.load(ftmat, allow_pickle=True)
                
                self.offset[i] = self.extra_args.get('extrapar')[i]['offset']
                
                self.M[i] = ResMLP(input_dim  = len(self.extra_args.get('ord')[i]), 
                                   output_dim = len(self.tmat[i]), 
                                   int_dim    = self.extra_args.get('extrapar')[i]['INTDIM'], 
                                   N_layer    = self.extra_args.get('extrapar')[i]['NLAYER'])
                self.M[i] = self.M[i].to('cpu')
                self.M[i] = nn.DataParallel(self.M[i])
                self.M[i].load_state_dict(torch.load(fname, map_location='cpu'))
                self.M[i] = self.M[i].module.to('cpu')
                self.M[i].eval()

                self.ord[i] = self.extra_args.get('ord')[i]
                self.req.extend(self.ord[i])
        
        self.req = list(set(self.req))
        d = {}
        for i in self.req:
            d[i] = None
        self.req = d

    def get_requirements(self):
        return self.req

    def predict(self, model, X, extrainfo, transform_matrix, offset):
        device   = 'cpu'
        X_mean   = torch.Tensor(extrainfo.item()['X_mean']).to(device)
        X_std    = torch.Tensor(extrainfo.item()['X_std']).to(device)
        Y_mean   = extrainfo.item()['Y_mean']
        Y_std    = extrainfo.item()['Y_std']
        Y_mean_2 = torch.Tensor(extrainfo.item()['Y_mean_2']).to(device)
        Y_std_2  = torch.Tensor(extrainfo.item()['Y_std_2']).to(device)
        X = torch.Tensor(X).to(device)
        with torch.no_grad():
            X_norm = ((X - X_mean) / X_std)
            X_norm.to(device)
            pred = model(X_norm)
            M_pred = pred.to(device)
            y_pred = (M_pred.float() *Y_std_2.float()+Y_mean_2.float()).cpu().numpy()
            y_pred = np.matmul(y_pred,transform_matrix)*Y_std+Y_mean
            y_pred = np.exp(y_pred) - offset
        return y_pred[0]
   
    def calculate(self, state, want_derived=True, **params):       
        par = params.copy()
        print(par)
        print(state)

        out    = ["dl","H"]
        idx    = np.where(np.array(self.extra_args.get('eval'))[:2])[0]
        for i in idx:
            params = self.extra_args.get('ord')[i]
            p = np.array([par[key] for key in params])
            state[out[i]] = self.predict(self.M[i], p, self.info[i], self.tmat[i], self.offset[i])

    def get_angular_diameter_distance(self, z):
        d_l = self.current_state["dl"].copy()
        d_a = d_l/(1.0 + self.z[0])**2
        D_A_interpolate = interpolate.interp1d(self.z[0], d_a)
        D_A = D_A_interpolate(z)
        try:
            l = len(D_A)
        except:
            D_A = np.array([D_A])
        else:
            l = 1
        return D_A

    def get_angular_diameter_distance_2(self, zpair):
        z_1, z_2 = zpair[0]
        
        if z_1 >= z_2:
            return 0
        else:
            da1 = self.get_angular_diameter_distance(z_1)
            da2 = self.get_angular_diameter_distance(z_2)
            cd1 = da1*(1+z_1)
            cd2 = da2*(1+z_2)
            return (cd2-cd1)/(1+z_2)

    def get_Hubble(self, z, units = "km/s/Mpc"):
        H = self.current_state["H"].copy()
        H_interpolate = interpolate.interp1d(self.z[1], H)
        H_arr = H_interpolate(z)
        try:
            l = len(H_arr)
        except:
            H_arr = np.array([H_arr])
        else:
            l = 1
        if units == "1/Mpc":
            H_arr /= 2.99792458e5
        return H_arr

    def get_can_support_params(self):
        return [ "omega_b", "omega_cdm", "h" ]
  