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
        RT = os.environ.get("ROOTDIR")
        self.lmax_theory = 9052
        self.ell         = np.arange(0,self.lmax_theory,1)
        # TT, TE, EE, PHIPHI, RD, THETA
        self.M     = [None, None, None, None, None, None, None, None, None]
        self.info  = [None, None, None, None, None, None, None, None, None]

        for i in range(3):
            if self.extra_args.get('eval')[i]:
                fname  = self.extra_args.get("file")[i]
                fextra = self.extra_args.get("extra")[i]
                self.info[i] = np.load(RT + "/" + fextra, allow_pickle=True)

                if self.extra_args.get('extrapar')[i]['MLA'] == 'TRF':
                    self.M[i] = TRF(input_dim=len(self.extra_args.get('ord')[i]),
                        output_dim=self.extra_args.get('extrapar')[i]['ellmax']-2,
                        int_dim=self.extra_args.get('extrapar')[i]['INTDIM'],
                        int_trf=self.extra_args.get('extrapar')[i]['INTTRF'],
                        N_channels=self.extra_args.get('extrapar')[i]['NCTRF'])
                elif self.extra_args.get('extrapar')[i]['MLA'] == 'CNN':
                    self.M[i] = CNNMLP(input_dim=len(self.extra_args.get('ord')[i]),
                                       output_dim=self.extra_args.get('extrapar')[i]['ellmax']-2,
                                       int_dim=self.extra_args.get('extrapar')[i]['INTDIM'],
                                       cnn_dim=self.extra_args.get('extrapar')[i]['INTCNN'])
                self.M[i] = self.M[i].to('cpu')
                self.M[i] = nn.DataParallel(self.M[i])
                self.M[i].load_state_dict(torch.load(RT + "/" + fname, map_location='cpu'))
                self.M[i] = self.M[i].module.to('cpu')
                self.M[i].eval()

        if self.extra_args.get('eval')[4]:
            fname  = self.extra_args.get("file")[4]
            fextra = self.extra_args.get("extra")[4]            
            self.info[4] = np.load(RT + "/" + fextra, allow_pickle=True)
            self.M[4]    = joblib.load(RT + "/" + fname)


        if self.extra_args.get('eval')[5]:
            fname  = self.extra_args.get("file")[5]
            fextra = self.extra_args.get("extra")[5]
            self.info[5] = np.load(RT + "/" + fextra,allow_pickle=True)
            self.M[5]    = joblib.load(RT + "/" + fname)


    def get_allow_agnostic(self):
        return True

    def predict_cmb(self,model,X, einfo):
        device = 'cpu'
        X_mean=torch.Tensor(einfo.item()['X_mean']).to(device)
        X_std=torch.Tensor(einfo.item()['X_std']).to(device)
        Y_mean=torch.Tensor(einfo.item()['Y_mean']).to(device)
        Y_std=torch.Tensor(einfo.item()['Y_std']).to(device)
        X = torch.Tensor(X).to(device)
        with torch.no_grad():
            X_norm = (X - X_mean)/X_std
            X_norm=torch.nan_to_num(X_norm, nan=0)
            X_norm.to(device)
            M_pred = model(X_norm).to(device)
        return (M_pred.float()*Y_std.float() + Y_mean.float()).cpu().numpy()
        
    def calculate(self, state, want_derived=False, **params):
        par = params.copy()

        # theta_star calculation begins ---------------------------
        if self.extra_args.get('eval')[5]:
            params = self.extra_args.get('ord')[5]
            p =  np.array([par[key] for key in params])-self.info[5].item()['X_mean']     
            par["H0"]=self.M[5].predict(p/self.info[5].item()['X_std'])[0]*self.info[5].item()['Y_std'][0]+self.info[5].item()['Y_mean'][0]
            par["omegam"]=(par["omegabh2"]+par["omegach2"])/(par["H0"]/100)**2+par["mnu"]*(3.046/3)**0.75/94.0708
            state["H0"]=par["H0"]
            state["omegam"]=par["omegam"]
        # theta_star calculation ends ---------------------------

        # cl calculation begins ---------------------------
        state["ell"] = self.ell.astype(int)
        state["bb"]  = np.zeros(self.lmax_theory)
        state["tt"]  = np.zeros(self.lmax_theory)
        state["te"]  = np.zeros(self.lmax_theory)
        state["ee"]  = np.zeros(self.lmax_theory)
        cmb  = ['tt', 'te', 'ee', 'pp']
        for i in range(3):
            if self.extra_args.get('eval')[i]:
                state[cmb[i]] = np.zeros(self.lmax_theory)
                params = self.extra_args.get('ord')[i]
                p = np.array([par[key] for key in params])
                logAs  = p[params.index('logA')]
                tau    = p[params.index('tau')]
                amp    = np.exp(logAs)/np.exp(2*tau)
                ellmax = self.extra_args.get('extrapar')[i]['ellmax']
                state[cmb[i]][2:ellmax] = self.predict_cmb(self.M[i], p, self.info[i])*amp
        state["et"] = state["te"]
        # cl calculation ends ---------------------------

        # rd calculation begins ---------------------------
        if self.extra_args.get('eval')[4]:
            params = self.extra_args.get('ord')[4]
            X_mean = self.info[4].item()['X_mean']
            Y_mean = self.info[4].item()['Y_mean']
            X_std  = self.info[4].item()['X_std']
            Y_std  = self.info[4].item()['Y_std']
            p  = np.array([par[key] for key in params]) - X_mean
            rd = self.M[4].predict(p/X_std)[0]*Y_std[0] + Y_mean[0]
            state['rdrag']   = rd
            state["derived"] = {"rdrag": rd}
        # cl calculation ends ---------------------------
        return True
    
    def get_can_provide_params(self):
        return ['H0','omegam','rdrag']

    def requested(self):
        return self._must_provide

    def get_H0(self):
        state = self.current_state.copy()
        return state["H0"]

    def get_omegam(self):
        state = self.current_state.copy()
        return state["omegam"]

    def get_rdrag(self):
        return self.M[4].predict(vd/self.info[4].item()['X_std'])[0]*self.info[4].item()['Y_std'][0]+self.info[4].item()['Y_mean'][0]

    def get_Cl(self, ell_factor = False, units = "1", unit_included = True, Tcmb=2.7255):
        cls_old = self.current_state.copy()
    
        cls_dict = {k : np.zeros(self.lmax_theory) for k in [ "tt", "te", "ee" , "et" , "bb" ]}
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