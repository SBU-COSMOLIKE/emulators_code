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

        # read the block from the YAML
        self.device = "cuda" if self.extra_args.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
        self.file = self.extra_args.get('file')[0]
        self.extra = self.extra_args.get('extra')[0]
        self.ord = self.extra_args.get('ord')[0]
        self.extrapar = self.extra_args.get('extrapar')[0]
        
        self.req = {} # construct the requirements
        for p in self.ord:
            self.req[p] = None

        # construct the network and load the weights
        self.model = ResTRF(input_dim = len(self.ord),
                            output_dim = self.extrapar['OUTPUT_DIM'],
                            int_dim_res = self.extrapar['INT_DIM_RES'],
                            int_dim_trf = self.extrapar['INT_DIM_TRF'],
                            N_channels = self.extrapar['NC_TRF'])
        self.model.to(self.device)
        state_dict = torch.load(self.file, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.M.append(self.model)

        # load extra stuff for
        if 'h5' in self.extra:
            with h5.File(self.extra, 'r') as f:
                self.samples_mean  = torch.Tensor(f['sample_mean'][:]).to(self.device)
                self.samples_std   = torch.Tensor(f['sample_std'][:]).to(self.device)
                self.dv_fid        = torch.Tensor(f['dv_fid'][:]).to(self.device)
                self.dv_evals      = torch.Tensor(f['dv_evals'][:]).to(self.device)
                self.dv_evecs      = torch.Tensor(f['dv_evecs'][:]).to(self.device)
                # right here I would like to have a "trained_params" 
                # so that we can check that ordering is correct!
                # we can save users from a simple mistake this way
        elif '.npy' in extra:
            # stuff, idk how its structured
            pass
        
        # invert the rotation matrix so that we don't do it every time we evaluate
        self.inv_dv_evecs = torch.linalg.inv(self.dv_evecs)

    def get_requirements(self):
        # remove shear calibration here and add to likelihood to make for cobaya speed hierarchy?
        return self.req

    def calculate(self, state, want_derived = True, **params):
        X = [params[p] for p in self.ord]
        with torch.no_grad():
            y_pred = self.M[0]((torch.Tensor(X).to(self.device) - self.samples_mean) / self.samples_std)
        y_pred = (y_pred * self.dv_evals) @ self.inv_dv_evecs + self.dv_fid
        state["cosmic_shear"] = y_pred[0].cpu().detach().numpy()
        return True

    def get_can_support_params(self):
        return ['cosmic_shear']

    def get_cosmic_shear(self):
        return self.current_state['cosmic_shear']




