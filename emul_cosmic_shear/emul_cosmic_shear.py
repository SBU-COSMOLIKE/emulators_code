import numpy as np
import os
import sys
import torch
import torch.nn as nn
from cobaya.theory import Theory
from cobaya.typing import InfoDict
import h5py as h5
sys.path.append(os.path.dirname(__file__))
from cobaya.theories.emul_cosmic_shear.emulator import ResTRF

class emul_cosmic_shear(Theory):
    extra_args: InfoDict = { }
    path: str

    def initialize(self):
        super().initialize()
        RT = os.environ.get("ROOTDIR")
        imax = 1
        for name in ("M", "info", "ord", "extrapar"):
            setattr(self, name, [None] * imax)
        for name in ("samples_mean", "samples_std", "dv_fid", "dv_evals", "dv_evecs", "inv_dv_evecs"):
            setattr(self, name, [None] * imax)
        self.req   = [] 
        self.device = "cuda" if self.extra_args.get("device") == "cuda" and torch.cuda.is_available() else "cpu"

        for i in range(imax):
            self.info[i] = self.extra_args.get('extra')[i]
            # load extra stuff for
            if 'h5' in self.info[i]:
                with h5.File(self.info[i], 'r') as f:
                    self.samples_mean[i] = torch.Tensor(f['sample_mean'][:]).to(self.device)
                    self.samples_std[i] = torch.Tensor(f['sample_std'][:]).to(self.device)
                    self.dv_fid[i] = torch.Tensor(f['dv_fid'][:]).to(self.device)
                    self.dv_evals[i] = torch.Tensor(f['dv_evals'][:]).to(self.device)
                    self.dv_evecs[i] = torch.Tensor(f['dv_evecs'][:]).to(self.device)
                    # right here I would like to have a "trained_params" 
                    # so that we can check that ordering is correct!
                    # we can save users from a simple mistake this way
            # invert the rotation matrix so that we don't do it every time we evaluate
            self.inv_dv_evecs[i] = torch.linalg.inv(self.dv_evecs[i])

            self.ord[i] = self.extra_args.get('ord')[i]
            self.extrapar[i] = self.extra_args["extrapar"][i].copy()

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

    def calculate(self, state, want_derived=True, **params):
        i = 0
        X = [params[p] for p in self.ord[i]]
        with torch.no_grad():
            y_pred = self.M[i]((torch.Tensor(X).to(self.device)-self.samples_mean[i])/self.samples_std[i])
        y_pred = (y_pred*self.dv_evals[i]) @ self.inv_dv_evecs[i] + self.dv_fid[i]
        state["cosmic_shear"] = y_pred[0].cpu().detach().numpy()
        return True

    def get_can_support_params(self):
        return ['cosmic_shear']

    def get_cosmic_shear(self):
        return self.current_state['cosmic_shear']




