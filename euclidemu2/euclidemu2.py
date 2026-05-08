"""
Cobaya theory that provides the non-linear correction to 
the matter perturbation using Euclid Emulator 2
"""

import torch, os, sys
import numpy as np
from cobaya.theory import Theory
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict
import euclidemu2 as ee2
from scipy.interpolate import interp1d, RectBivariateSpline

class euclidemu2(Theory):
    renames: Mapping[str, str] = empty_dict
    extra_args: InfoDict = { }
    _must_provide: dict
    path: str

    def initialize(self):
        super().initialize()
        ROOTDIR = os.environ.get("ROOTDIR")

        # Accuracy boost here works just like cosmolike accuracy:
        #   It sets the density of k and z to compute EE2.
        self.accuracyboost = self.extra_args['accuracyboost']

        tmp=int(min(120 + 20*self.accuracyboost,250))

        self.z = np.concatenate((np.linspace(0,3.0,max(50,int(0.75*tmp))), 
            np.linspace(3.01,9.99,max(10,int(0.25*tmp)))),axis=0)
        self.len_z = len(self.z)

        self.log10k = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost)) # k/h
        self.k = 10**self.log10k
        self.len_k = len(self.k)

        self.emulator = ee2.PyEuclidEmulator()

        # Lambda functions for derived quantities
        self.calc_omegam = lambda omegabh2, omegach2, h, mnu: (omegabh2 + omegach2 + (mnu*(3.046/3)**0.75)/94.0708)/(h*h)
        self.calc_omegab = lambda omegabh2, h: omegabh2/(h*h)

    def get_requirements(self):
        return {
            "omegabh2": None,
            "omegach2": None,
            "As": None,
            "ns": None,
            "mnu": None, # Needed to convert to omegam
            "H0": None,
            "w": None,
            "wa": None
        }

    def get_can_provide(self):
        return {'non_linear_ratio': None}

    def calculate(self, state, want_derived=True, **params):
        # get params that go to additional computations
        # NOTE: CAMB requires omegabh2 and omegach2, provides Omegam
        # Therefore this theory cannot require Omegam
        h        = self.provider.get_param("H0")/100
        omegabh2 = self.provider.get_param('omegabh2')
        omegach2 = self.provider.get_param('omegach2')
        mnu      = self.provider.get_param('mnu')

        # derived quantities
        omegab = self.calc_omegab(omegabh2, h)
        omegam = self.calc_omegam(omegabh2, omegach2, h, mnu)

        # get the boost
        params = {
          'Omm'  : omegam,
          'As'   : self.provider.get_param("As"),
          'Omb'  : omegab,
          'ns'   : self.provider.get_param("ns"),
          'h'    : h,
          'mnu'  : mnu, 
          'w'    : self.provider.get_param("w"),
          'wa'   : self.provider.get_param("wa"),
        }

        kbt, tmp_bt = ee2.get_boost2(params, 
                                     self.z, 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_k))  
        logbt = np.log(np.array(tmp_bt, dtype='float64'))

        # We need to do interpolation for extrapolation
        # Two approaches:
        ########################################################################
        
        # Evan's version: constant extrapolation (this is what EE2 natively does)
        # Using LSST_Y1/EXAMPLE_EVALUATE1.yaml
        #   Delta_chi2 with original version: 0.2
        #   Delta_chi2 with Halofit: 11.5
        boost = np.exp(interp1d(np.log10(kbt), 
                        logbt, 
                        axis=1,
                        kind='linear', 
                        bounds_error=False,
                        fill_value=(logbt[:,0], logbt[:,-1]), # constant extrapolation
                        assume_sorted=True)(self.log10k))
        
        # # Original version: linear extrap for high k, boost=1.0 for low k
        # # Using LSST_Y1/EXAMPLE_EVALUATE1.yaml
        # #   Delta_chi2 with Halofit: 11.7
        # boost = np.exp(interp1d(np.log10(kbt), 
        #                 logbt, 
        #                 axis=1,
        #                 kind='linear', 
        #                 fill_value='extrapolate',
        #                 assume_sorted=True)(self.log10k_interp_2D))
        # boost[:, self.log10k_interp_2D<-2.0589] = 1.0

        # update cobaya state with result
        state["non_linear_ratio"] = {
            'k_h':   self.k/h,
            'z':     self.z,
            'ratio': np.sqrt(boost)
        }

    def get_non_linear_ratio(self, results):
        return self.current_state["non_linear_ratio"]
