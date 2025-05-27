import numpy as np
import os
import sys
import torch
import torch.nn as nn
from cobaya.theory import Theory
from cobaya.typing import InfoDict
import h5py as h5

sys.path.append(os.path.dirname(__file__))
from cocoa_emu import nn_emulator


class lsst_y1_cosmic_shear_emulator(Theory):
    extra_args: InfoDict = { }

    def initialize(self):
        super().initialize()

        PATH = os.environ.get("ROOTDIR") + "/" + self.extra_args.get('filename')

        self.model = nn_emulator.nn_emulator(preset='xi_restrf')
        self.model.load(PATH)

        self.shear_calib_mask = np.load(os.environ.get("ROOTDIR") + '/external_modules/data/lsst_y1_cosmic_shear_emulator/shear_calib_mask.npy')[:,:780] 

    def add_shear_calib(self, m, datavector):
        for i in range(5):
            factor = (1 + m[i])**self.shear_calib_mask[i]
            factor = factor[0:780]
            datavector = factor * datavector
        return datavector

    def get_requirements(self):
        # remove shear calibration here and add to likelihood to make for cobaya speed hierarchy?
        return {
          "logA": None,
          "H0": None,
          "ns": None,
          "omegabh2": None,
          "omegach2": None,
          "LSST_DZ_S1": None,
          "LSST_DZ_S2": None,
          "LSST_DZ_S3": None,
          "LSST_DZ_S4": None,
          "LSST_DZ_S5": None,
          "LSST_A1_1": None,
          "LSST_A1_2": None,
          "LSST_M1": None,
          "LSST_M2": None,
          "LSST_M3": None,
          "LSST_M4": None,
          "LSST_M5": None
        }

    def calculate(self, state, want_derived = True, **params):
        xi_params = [
            params['logA'],params['ns'],params['H0'],params['omegabh2'],params['omegach2'],
            params['LSST_DZ_S1'],params['LSST_DZ_S2'],params['LSST_DZ_S3'],params['LSST_DZ_S4'],params['LSST_DZ_S5'],
            params['LSST_A1_1'],params['LSST_A1_2'] 
        ]

        shear_calibration_params = [
            params['LSST_M1'],params['LSST_M2'],params['LSST_M3'],params['LSST_M4'],params['LSST_M5']
        ]

        xi_pm = self.model.predict(xi_params)[0]
        xi_pm = self.add_shear_calib(shear_calibration_params, xi_pm)

        state["lsst_y1_xi"] = xi_pm

        return True

    def get_can_support_params(self):
        return ['lsst_y1_xi']

    def get_lsst_y1_xi(self):
        return self.current_state['lsst_y1_xi']




