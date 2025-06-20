import numpy as np
import os, joblib

class emultheta():    
    def __init__(self, extra_args):
        self.extra_args = extra_args
        RT = os.environ.get("ROOTDIR")
        self.M      = [None]
        self.info   = [None]
        self.ord    = [None]

        fname  = RT + "/" + self.extra_args.get("file")[0]
        fextra = RT + "/" + self.extra_args.get("extra")[0]
        self.info[0] = np.load(fextra,allow_pickle=True)
        self.M[0]    = joblib.load(fname)
        self.ord[0]  = self.extra_args.get('ord')[0]

    def calculate(self, par):
        state = {}
        X_mean = self.info[0].item()['X_mean']
        Y_mean = self.info[0].item()['Y_mean']
        X_std  = self.info[0].item()['X_std']
        Y_std  = self.info[0].item()['Y_std']
        p =  np.array([par[key] for key in self.ord[0]]) - X_mean     
        H0 = self.M[0].predict(p/X_std)[0]*Y_std[0] + Y_mean[0]
        state.update({"H0": H0})
        state.update({"omegam": par["omegamh2"]/(H0/100.0)**2})
        return state

    def get_H0(self, params):
        state = self.calculate(params)
        return state["H0"]

    def get_omegam(self, params):
        state = self.calculate(params)
        return state["omegam"]