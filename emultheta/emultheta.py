import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cobaya.theory import Theory
from cobaya.typing import InfoDict
from cobaya.theories.emulcmb.emulator import Supact, Affine, Better_Attention, Better_Transformer, ResBlock, ResMLP, TRF, CNNMLP, ResMLP2
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
        imax = 1
        for name in ("M", "info", "ord", "extrapar"):
            setattr(self, name, [None] * imax )
        self.req    = [] 
        self.device = self.extra_args.get("device")
        if self.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # BASIC CHECKS BEGINS ------------------------------------------------
        _required_lists = [
            ("extra", "Emulator Theta: Missing emulator file (extra) option"),
            ("ord", "Emulator Theta: Missing ord (parameter ordering) option"),
            ("file", "Emulator Theta: Missing emulator file option"),
            ("extrapar", "Emulator Theta: Missing extrapar option"),
        ]
        for key, msg in _required_lists:
            if (tmp := self.extra_args.get(key)) is None:
                raise ValueError(msg)        
            if any(x is None or (isinstance(x, str) and x.strip().lower() == "none") for x in tmp[:imax]):
                raise ValueError(msg)
        _mla_requirements = {
            "GP":  [],
            "ResMLP": ["INTDIM","NLAYER"],
        }
        params = self.extra_args["extrapar"][0]
        if not isinstance(params, dict):
            raise ValueError('Emulator Theta: extrapar option not a dictionary')
        mla = params.get('MLA')
        if mla is None or (isinstance(mla, str) and mla.strip().lower() == "none"):
            raise ValueError(f'Emulator Theta: Missing extrapar MLA option')
        try:
            req_keys = _mla_requirements[mla]
        except KeyError:
            raise KeyError(f"Emulator Theta: Unknown MLA option: {mla}")
        miss = [k for k in req_keys if k not in params]
        if miss:
            raise KeyError(f"Emulator Theta: Missing extrapar keys for {mla}: {miss}")
        # BASIC CHECKS ENDS ------------------------------------------------
        
        fname  = RT + "/" + self.extra_args.get("file")[0]
        fextra = RT + "/" + self.extra_args.get("extra")[0]
        
        self.info[0] = np.load(fextra,allow_pickle=True)
        self.ord[0]  = self.extra_args.get('ord')[0]

        self.MLA = self.extra_args['extrapar'][0]['MLA']
        if self.MLA == "GP":
            self.M[0]    = joblib.load(fname)
        elif self.MLA == "ResMLP":
            intdim = self.extra_args.get('extrapar')[0]['INTDIM']
            Nlayer = self.extra_args.get('extrapar')[0]['NLAYER']
            self.M[0] = ResMLP2(input_dim=len(self.ord[0]),output_dim=1,int_dim=intdim,N_layer=Nlayer)
            self.M[0] = self.M[0].to(self.device)
            self.M[0] = nn.DataParallel(self.M[0])
            self.M[0].load_state_dict(torch.load(fname, map_location=self.device))
            self.M[0] = self.M[0].module.to(self.device)
            self.M[0].eval()
        self.req.extend(self.ord[0])

        self.req = list(set(self.req))
        d = {'omegamh2' : None}
        for i in self.req:
            d[i] = None
        self.req = d

    def get_requirements(self):
        return self.req

    def predict(self, X, Y_mean, Y_std, model):
        with torch.no_grad():
            X = torch.Tensor(X).to(self.device)
            
            pred = model(X)
            
            M_pred = pred.to(self.device).float().cpu().numpy()
            y_pred = (M_pred *Y_std+Y_mean)
            
        return y_pred

    def calculate(self, state, want_derived=False, **params):
        par = params.copy()

        params = self.ord[0]
        X_mean = self.info[0].item()['X_mean']
        Y_mean = self.info[0].item()['Y_mean']
        X_std  = self.info[0].item()['X_std']
        Y_std  = self.info[0].item()['Y_std']
        p =  np.array([par[key] for key in params]) - X_mean
        if self.MLA == "GP":
            H0 = self.M[0].predict(p/X_std)[0]*Y_std[0] + Y_mean[0]
        elif self.MLA == "ResMLP":
            p0 = p/X_std
            H0 = self.predict(p0, Y_mean, Y_std, self.M[0])[0][0]
        
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
 