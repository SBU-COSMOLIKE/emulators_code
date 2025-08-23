import numpy as np
import os, joblib
import torch
import torch.nn as nn
from cobaya.theories.emultheta.emulator import ResMLP2
class emultheta():    
    def __init__(self, extra_args):
        self.extra_args = extra_args
        RT = os.environ.get("ROOTDIR")
        imax = 1
        for name in ("M", "info", "ord", "extrapar"):
            setattr(self, name, [None] * imax)
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
        self.extrapar = self.extra_args["extrapar"][0]
        if not isinstance(self.extrapar, dict):
            raise ValueError('Emulator Theta: extrapar option not a dictionary')
        self.MLA = self.extrapar.get('MLA')
        if self.MLA is None or (isinstance(self.MLA, str) and self.MLA.strip().lower() == "none"):
            raise ValueError(f'Emulator Theta: Missing extrapar MLA option')
        try:
            req_keys = _mla_requirements[self.MLA]
        except KeyError:
            raise KeyError(f"Emulator Theta: Unknown MLA option: {self.MLA}")
        miss = [k for k in req_keys if k not in self.extrapar]
        if miss:
            raise KeyError(f"Emulator Theta: Missing extrapar keys for {self.MLA}: {miss}")
        # BASIC CHECKS ENDS ------------------------------------------------

        file = os.path.join(RT, self.extra_args.get("extra")[0])
        self.info[0] = np.load(file,allow_pickle=True)
        self.ord[0]  = self.extra_args.get('ord')[0]
        
        if self.MLA == "GP":
            file = os.path.join(RT, self.extra_args.get("file")[0])
            self.M[0] = joblib.load(file)
        elif self.MLA == "ResMLP":
            self.M[0] = ResMLP2(input_dim = len(self.ord[0]),
                                output_dim = 1,
                                int_dim = self.extrapar['INTDIM'],
                                N_layer = self.extrapar['NLAYER'])
            self.M[0] = self.M[0].to(self.device)
            self.M[0] = nn.DataParallel(self.M[0])
            file = os.path.join(RT, self.extra_args.get("file")[0])
            self.M[0].load_state_dict(torch.load(file, map_location=self.device))
            self.M[0] = self.M[0].module.to(self.device)
            self.M[0].eval()

    def predict(self, X, Y_mean, Y_std, model):
        with torch.no_grad():
            X = torch.Tensor(X).to(self.device)
            
            pred = model(X)
            
            M_pred = pred.to(self.device).float().cpu().numpy()
            y_pred = (M_pred *Y_std+Y_mean)
            
        return y_pred
        

    def calculate(self, par):


        state = {}
        X_mean = self.info[0].item()['X_mean']
        Y_mean = self.info[0].item()['Y_mean']
        X_std  = self.info[0].item()['X_std']
        Y_std  = self.info[0].item()['Y_std']
        p =  np.array([par[key] for key in self.ord[0]]) - X_mean
        if self.MLA == "GP":
            H0 = self.M[0].predict(p/X_std)[0]*Y_std[0] + Y_mean[0]
        elif self.MLA == "ResMLP":
            p0 = p/X_std
            H0 = self.predict(p0, Y_mean, Y_std, self.M[0])[0][0]
        
        h2       = (H0/100.0)**2
        omegamh2 = par["omegamh2"]

        state.update({"H0": H0})
        state.update({"omegam": omegamh2/h2})    
   
        return state

    def get_H0(self, params):
        state = self.calculate(params)
        return state["H0"]

    def get_omegam(self, params):
        state = self.calculate(params)
        return state["omegam"]