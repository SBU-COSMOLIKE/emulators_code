import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cobaya.theory import Theory
#from cobaya.theories.cosmo import BoltzmannBase
from cobaya.typing import InfoDict
from cobaya.theories.emulcmb.emulator import Supact, Affine, Better_Attention, Better_Transformer, ResBlock, ResMLP, TRF, CNNMLP
import joblib
import scipy
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict

class emulcmb(Theory):
    renames: Mapping[str, str] = empty_dict
    extra_args: InfoDict = { }
    _must_provide: dict
    path: str
    
    def initialize(self):
        super().initialize()
        self.ordering = self.extra_args.get('ordering')
        self.thetaordering = 0.0#self.extra_args.get('thetaordering')
        
        self.ROOT = os.environ.get("ROOTDIR")

        self.PATH1 = self.ROOT + "/" + self.extra_args.get('ttfilename')
        self.PATH2 = self.ROOT + "/" + self.extra_args.get('tefilename')
        self.PATH3 = self.ROOT + "/" + self.extra_args.get('eefilename')

        
        self.PATH7 = "INIT"  # File that contains GP model for theta to H0

        self.extrainfo_TT = np.load(self.ROOT+"/"+self.extra_args.get('ttextraname'),allow_pickle=True)
        self.extrainfo_TE = np.load(self.ROOT+"/"+self.extra_args.get('teextraname'),allow_pickle=True)
        self.extrainfo_EE = np.load(self.ROOT+"/"+self.extra_args.get('eeextraname'),allow_pickle=True)
        
        self.extrainfo_GP = 0.0 # extra info file for GP of theta to H0
        
        intdim = 4
        nlayer = 4
        nc = 16
        inttrf=5120
        device = 'cpu'
        intdim_simple = 1
        nlayer_simple = 1
        cnn_dim = 3200

        self.model_type = self.extra_args.get('modeltype')
        self.ell_max = self.extra_args.get('ellmax')

        if self.model_type == 'TRF':
            self.model1 = TRF(input_dim=len(self.ordering),output_dim=self.ell_max-2, int_dim=intdim, int_trf=inttrf,N_channels=nc)
            self.model2 = TRF(input_dim=len(self.ordering),output_dim=self.ell_max-2, int_dim=intdim, int_trf=inttrf,N_channels=nc)
            self.model3 = TRF(input_dim=len(self.ordering),output_dim=self.ell_max-2, int_dim=intdim, int_trf=inttrf,N_channels=nc)


        elif self.model_type == 'CNN':

            self.model1 = CNNMLP(input_dim=len(self.ordering),output_dim=self.ell_max-2,int_dim=intdim, cnn_dim=cnn_dim)
            self.model2 = CNNMLP(input_dim=len(self.ordering),output_dim=self.ell_max-2,int_dim=intdim, cnn_dim=cnn_dim)
            self.model3 = CNNMLP(input_dim=len(self.ordering),output_dim=self.ell_max-2,int_dim=intdim, cnn_dim=cnn_dim)

        else:
            print('No model')

        self.model7 = 0.0 # load GP model for theta to H0

        self.model1 = self.model1.to(device)
        self.model2 = self.model2.to(device)
        self.model3 = self.model3.to(device)

        self.model1 = nn.DataParallel(self.model1)
        self.model2 = nn.DataParallel(self.model2)
        self.model3 = nn.DataParallel(self.model3)

        self.model1.load_state_dict(torch.load(self.PATH1+'.pt',map_location=device))
        self.model2.load_state_dict(torch.load(self.PATH2+'.pt',map_location=device))
        self.model3.load_state_dict(torch.load(self.PATH3+'.pt',map_location=device))
        

        self.model1 = self.model1.module.to(device)
        self.model2 = self.model2.module.to(device)
        self.model3 = self.model3.module.to(device)
        
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()


        self.ell = np.arange(0,9052,1)
        self.lmax_theory = 9052

        self.testh0 = -1

    def get_allow_agnostic(self):
        return True

    def predict(self,model,X, extrainfo):
        device = 'cpu'
        
        X_mean=torch.Tensor(extrainfo.item()['X_mean']).to(device)
        
        X_std=torch.Tensor(extrainfo.item()['X_std']).to(device)
        
        Y_mean=torch.Tensor(extrainfo.item()['Y_mean']).to(device)
        
        Y_std=torch.Tensor(extrainfo.item()['Y_std']).to(device)
    
        X = torch.Tensor(X).to(device)

        with torch.no_grad():
            X_norm = (X - X_mean)/X_std
            X_norm=torch.nan_to_num(X_norm, nan=0)
            X_norm.to(device)
            M_pred = model(X_norm).to(device)
        return (M_pred.float()*Y_std.float() + Y_mean.float()).cpu().numpy()

    def scaletrans(self,y_pred,X):
        return y_pred*np.exp(X[self.ordering.index('logA')])/np.exp(2*X[self.ordering.index('tau')])
        
    def calculate(self, state, want_derived=False, **params):
        cmb_param = params.copy()

        if 'H0' not in cmb_param:
            if self.testh0 < 0:
                # This is the file that contains GP model for theta to H0
                self.PATH7 = self.ROOT + "/" + self.extra_args.get('GPfilename')
                self.thetaordering = self.extra_args.get('thetaordering')

                # load GP model for theta to H0
                self.model7 = joblib.load(self.PATH7) 

                self.extrainfo_GP = np.load(self.ROOT + 
                                            "/"  + 
                                            self.extra_args.get('GPextraname'), 
                                                                allow_pickle=True)
            self.testh0 = 1



            vt =  np.array([cmb_param[key] for key in self.thetaordering]) - self.extrainfo_GP.item()['X_mean']
                    
            cmb_param["H0"]= self.model7.predict(vt/self.extrainfo_GP.item()['X_std'])[0]*self.extrainfo_GP.item()['Y_std'][0] + self.extrainfo_GP.item()['Y_mean'][0]

        cmb_params = np.array([cmb_param[key] for key in self.ordering])

        state["ell"] = self.ell.astype(int)
        state["tt"] = np.zeros(self.lmax_theory)
        state["te"] = np.zeros(self.lmax_theory)
        state["bb"] = np.zeros(self.lmax_theory)
        state["ee"] = np.zeros(self.lmax_theory)
        state["tt"][2:self.ell_max] = self.scaletrans(self.predict(self.model1,
                                                           cmb_params,
                                                           self.extrainfo_TT), 
                                              cmb_params)[0]
        state["te"][2:self.ell_max] = self.scaletrans(self.predict(self.model2, 
                                                           cmb_params, 
                                                           self.extrainfo_TE), 
                                              cmb_params)[0]
        state["ee"][2:self.ell_max] = self.scaletrans(self.predict(self.model3, 
                                                           cmb_params, 
                                                           self.extrainfo_EE), 
                                              cmb_params)[0]
        state["et"] = state["te"]

        return True

    def get_Cl(self, ell_factor = False, units = "1", unit_included = True, Tcmb=2.7255):
        cls_old = self.current_state.copy()
        
        cls_dict = { k : np.zeros(self.lmax_theory) for k in [ "tt", "te", "ee" , "et" , "bb" ] }
        cls_dict["ell"] = self.ell
        
        ls = self.ell
        
        if ell_factor:
            for k in [ "tt", "te", "ee" , "et" , "bb"]:
                ls_fac = self.ell_factor(ls,k)
                cls_dict[k] = cls_old[k] * ls_fac
        else:
            for k in [ "tt", "te", "ee" , "et" , "bb"]:
                cls_dict[k] = cls_old[k]
        if unit_included:
            unit=1
        else:
            for k in [ "tt", "te", "ee" , "et" , "bb" ]:
            
                unit = self.cmb_unit_factor(k, units, Tcmb)
            
                cls_dict[k] = cls_dict[k] * unit
        
        return cls_dict
        
    def ell_factor(self, ls: np.ndarray, spectra: str) -> np.ndarray:
        """
        Calculate the ell factor for a specific spectrum.
        These prefactors are used to convert from Cell to Dell and vice-versa.

        See also:
        cobaya.BoltzmannBase.get_Cl
        `camb.CAMBresults.get_cmb_power_spectra
        <https://camb.readthedocs.io/en/latest/results.html#camb.results.CAMBdata.get_cmb_power_spectra>`_

        Examples:

        ell_factor(l, "tt") -> :math:`\ell ( \ell + 1 )/(2 \pi)`

        ell_factor(l, "pp") -> :math:`\ell^2 ( \ell + 1 )^2/(2 \pi)`.

        :param ls: the range of ells.
        :param spectra: a two-character string with each character being one of [tebp].

        :return: an array filled with ell factors for the given spectrum.
        """
        ellfac = np.ones_like(ls).astype(float)

        if spectra in ["tt", "te", "tb", "ee", "et", "eb", "bb", "bt", "be"]:
            ellfac = ls * (ls + 1.0) / (2.0 * np.pi)
        elif spectra in ["pt", "pe", "pb", "tp", "ep", "bp"]:
            ellfac = (ls * (ls + 1.0)) ** (3. / 2.) / (2.0 * np.pi)
        elif spectra in ["pp"]:
            ellfac = (ls * (ls + 1.0)) ** 2.0 / (2.0 * np.pi)

        return ellfac

    def cmb_unit_factor(self, spectra: str,
                        units: str = "1",
                        Tcmb: float = 2.7255) -> float:
        """
        Calculate the CMB prefactor for going from dimensionless power spectra to
        CMB units.

        :param spectra: a length 2 string specifying the spectrum for which to
                        calculate the units.
        :param units: a string specifying which units to use.
        :param Tcmb: the used CMB temperature [units of K].
        :return: The CMB unit conversion factor.
        """
        res = 1.0
        
        x, y = spectra.lower()

        if x == "t" or x == "e" or x == "b":
            res *= self._cmb_unit_factor(units, Tcmb)
        elif x == "p":
            res *= 1. / np.sqrt(2.0 * np.pi)

        if y == "t" or y == "e" or y == "b":
            res *= self._cmb_unit_factor(units, Tcmb)
        elif y == "p":
            res *= 1. / np.sqrt(2.0 * np.pi)
        
        return res
    
    def get_can_support_params(self):
        return [ "omega_b", "omega_cdm", "h", "logA", "ns", "tau_reio" ]
    