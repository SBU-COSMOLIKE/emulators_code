import numpy as np
import os
import sys
import torch
import torch.nn as nn
from cobaya.theory import Theory
from cobaya.typing import InfoDict
import h5py as h5

sys.path.append(os.path.dirname(__file__))
from emulator import ResTRF

class cosmic_shear(Theory):
    extra_args: InfoDict = { }
    path: str

    def initialize(self):
        super().initialize()

        #PATH = os.environ.get("ROOTDIR") + "/" + self.extra_args.get('filename')
        RT = os.environ.get("ROOTDIR")

        # read the block from the YAML
        self.device      = self.extra_args.get('device')
        self.file        = self.extra_args.get('file')[0]
        self.extra       = self.extra_args.get('extra')[0]
        self.ord         = self.extra_args.get('ord')[0]
        self.fast_params = self.extra_args.get('fast_params')[0]
        self.extrapar    = self.extra_args.get('extrapar')[0]

        self.M = []

        # construct the requirements
        req_params = np.concatenate((self.ord,self.fast_params))

        self.req = {}

        for p in req_params:
            self.req[p] = None

        # set the model to load the device, check if cuda is really available and set tensor type
        if self.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_device(self.device)

        # construct the network and load the weights
        self.model = ResTRF(
                        input_dim = len(self.ord),
                        output_dim = self.extrapar['OUTPUT_DIM'],
                        int_dim_res = self.extrapar['INT_DIM_RES'],
                        int_dim_trf = self.extrapar['INT_DIM_TRF'],
                        N_channels = self.extrapar['NC_TRF']
        )
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

        # for fast parameters
        self.shear_calib_mask = np.load(RT + '/external_modules/data/cosmic_shear/shear_calib_mask.npy')[:,:780] 

    def add_shear_calib(self, m, datavector):
        # this is to get a working datavector calculation in the emulator.
        # we need to figure out how to apply shear calibration without the mask
        # or how to generate a mask.

        for i in range(5):
            factor = (1 + m[i])**self.shear_calib_mask[i]
            factor = factor[0:780]
            datavector = factor * datavector
        return datavector

    def get_requirements(self):
        # remove shear calibration here and add to likelihood to make for cobaya speed hierarchy?
        return self.req

    def calculate(self, state, want_derived = True, **params):
        X = [params[p] for p in self.ord]
        M = [params[p] for p in self.fast_params]

        with torch.no_grad():
            y_pred = self.M[0]((torch.Tensor(X).to(self.device) - self.samples_mean) / self.samples_std)

        y_pred = (y_pred * self.dv_evals) @ self.inv_dv_evecs + self.dv_fid

        state["lsst_y1_xi"] = self.add_shear_calib(M,y_pred[0].cpu().detach().numpy())
        np.savetxt('/home/grads/extra_data/evan/cocoa/Cocoa/state.txt',state["lsst_y1_xi"])

        return True

    def get_can_support_params(self):
        return ['lsst_y1_xi']

    def get_lsst_y1_xi(self):
        return self.current_state['lsst_y1_xi']




