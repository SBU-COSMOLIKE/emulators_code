import numpy as np
import os, joblib
import os.path as path

class emulrdrag():    
    def __init__(self, extra_args):
        self.extra_args = extra_args
        RT = os.environ.get("ROOTDIR")
        self.imax = 1
        for name in ("M", "info", "ord"):
            setattr(self, name, [None] * self.imax)

        # BASIC CHECKS BEGINS ------------------------------------------------
        _required_lists = [
            ("extra", "Emulator Theta: Missing emulator file (extra) option"),
            ("ord", "Emulator Theta: Missing ord (parameter ordering) option"),
            ("file", "Emulator Theta: Missing emulator file option")
        ]
        for key, msg in _required_lists:
            if (tmp := self.extra_args.get(key)) is None:
                raise ValueError(msg)        
            if any(x is None or (isinstance(x,str) and 
                   x.strip().lower() == "none") for x in tmp[:self.imax]):
                raise ValueError(msg)
        # BASIC CHECKS ENDS ------------------------------------------------
        
        self.info[0] = np.load(path.join(RT,self.extra_args.get("extra")[0]),allow_pickle=True)        
        self.M[0] = joblib.load(path.join(RT,self.extra_args.get("file")[0]))
        self.ord[0] = self.extra_args.get('ord')[0]

    def calculate(self, par):
        state = {}
        X_mean = self.info[0].item()['X_mean']
        Y_mean = self.info[0].item()['Y_mean']
        X_std  = self.info[0].item()['X_std']
        Y_std  = self.info[0].item()['Y_std']
        p =  np.array([par[key] for key in self.ord[0]]) - X_mean     
        rd = self.M[0].predict(p/X_std)[0]*Y_std[0] + Y_mean[0]
        state.update({'rdrag':rd})
        
        return state

    def get_rdrag(self, params):
        state = self.calculate(params)
        return state["rdrag"] 