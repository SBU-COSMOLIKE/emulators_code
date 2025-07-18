import torch, os, sys
import torch.nn as nn
import numpy as np
from cobaya.theory import Theory
from cobaya.theories.wtheta.emulator import ResTRF
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict
import h5py as h5
sys.path.append(os.path.dirname(__file__))

class emul_gg(Theory):
    renames: Mapping[str, str] = empty_dict
    extra_args: InfoDict = { }
    _must_provide: dict
    path: str

    def initialize(self):
        super().initialize()
        RT = os.environ.get("ROOTDIR")
        imax = 1
        self.eval  = [True] * imax
        for name in ("M", "info", "ord", "extrapar"):
            setattr(self, name, [None] * imax)
        for name in ("X_mean", "X_std", "Y_mean", "dv_evals", "dv_evecs", "inv_dv_evecs"):
            setattr(self, name, [None] * imax)
        self.req   = [] 
        self.device = "cuda" if self.extra_args.get("device") == "cuda" and torch.cuda.is_available() else "cpu"

        # BASIC CHECKS BEGINS ------------------------------------------------
        _required_lists = [
            ("extra", "Emulator wtheta: Missing emulator file (extra) option"),
            ("ord", "Emulator wtheta: Missing ord (parameter ordering) option"),
            ("file", "Emulator wtheta: Missing emulator file option"),
            ("extrapar", "Emulator wtheta: Missing extrapar option"),
        ]
        for key, msg in _required_lists:
            if (tmp := self.extra_args.get(key)) is None or (len(tmp)<imax):
                raise ValueError(msg)        
            if any(self.eval[i] and 
                   (tmp[i] is None or (isinstance(tmp[i], str) and tmp[i].strip().lower()=="none")) 
                   for i in range(imax)):
                raise ValueError(msg)
        for i in range(imax):
            if not self.eval[i]:
                continue
            self.extrapar[i] = self.extra_args["extrapar"][i].copy()
            if not isinstance(self.extrapar[i], dict):
                raise ValueError('Emulator wtheta: extrapar option not a dictionary')
        # BASIC CHECKS ENDS ------------------------------------------------
        
        for i in range(imax):
            self.info[i] = self.extra_args.get('extra')[i]
            # load extra stuff for
            if 'h5' in self.info[i]:
                with h5.File(self.info[i], 'r') as f:
                    self.X_mean[i] = torch.Tensor(f['sample_mean'][:]).to(self.device)
                    self.X_std[i] = torch.Tensor(f['sample_std'][:]).to(self.device)
                    self.Y_mean[i] = torch.Tensor(f['dv_fid'][:]).to(self.device)
                    self.dv_evals[i] = torch.Tensor(f['dv_evals'][:]).to(self.device)
                    self.dv_evecs[i] = torch.Tensor(f['dv_evecs'][:]).to(self.device)
                    # right here I would like to have a "trained_params" 
                    # so that we can check that ordering is correct!
                    # we can save users from a simple mistake this way
            # invert the rotation matrix so that we don't do it every time we evaluate
            self.inv_dv_evecs[i] = torch.linalg.inv(self.dv_evecs[i])

            self.ord[i] = self.extra_args.get('ord')[i]

            if self.extrapar[i]['MLA'] == 'TRF':
                self.M[i] = ResTRF(input_dim = len(self.ord[i]),
                                   output_dim  = self.extrapar[i]['OUTPUT_DIM'],
                                   int_dim_res = self.extrapar[i]['INT_DIM_RES'],
                                   int_dim_trf = self.extrapar[i]['INT_DIM_TRF'],
                                   N_channels  = self.extrapar[i]['NC_TRF'])
            self.M[i] = self.M[i].to(self.device)
            self.M[i].load_state_dict(torch.load(self.extra_args.get('file')[i],map_location=self.device))
            self.M[i] = self.M[i].eval()
            self.req.extend(self.ord[i])

        self.req = list(set(self.req))
        d = {}
        for i in self.req:
            d[i] = None
        self.req = d

    def get_requirements(self):
        return self.req

    def predict_data_vector(self, X, i):
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            X_norm = torch.nan_to_num((X-self.X_mean[i])/self.X_std[i],nan=0).to(self.device)
            M_pred = self.M[i](X_norm).to(self.device)
        res = (M_pred*self.dv_evals[i]) @ self.inv_dv_evecs[i] + self.Y_mean[i]
        return res[0].cpu().detach().numpy()

    def calculate(self, state, want_derived=True, **params):
        i = 0
        X = [params[p] for p in self.ord[i]]
        state["wtheta"] = self.predict_data_vector(X,i)
        return True

    def get_can_support_params(self):
        return ['wtheta']

    def get_wtheta(self):
        return self.current_state['wtheta']




