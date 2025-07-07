import torch, os
import torch.nn as nn
import numpy as np
from cobaya.theories.emulcmb.emulator import ResBlock, ResMLP, TRF, CNNMLP
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict

class emulcmb():    
    def __init__(self, extra_args):
        self.extra_args = extra_args
        RT = os.environ.get("ROOTDIR")
        self.lmax_theory = 9052
        self.ell         = np.arange(0,self.lmax_theory,1)
        # TT, TE, EE, PHIPHI 
        self.eval   = [False, False, False, False]
        self.M      = [None, None, None, None]
        self.info   = [None, None, None, None]
        self.ord    = [None, None, None, None]
        self.device = self.extra_args.get("device")
        if self.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        for i in range(4):
            if self.extra_args.get('eval')[i]:
                fname  = RT + "/" + self.extra_args.get("file")[i]
                fextra = RT + "/" + self.extra_args.get("extra")[i]
                self.info[i] = np.load(fextra, allow_pickle=True)

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
                elif self.extra_args.get('extrapar')[i]['MLA'] == 'ResMLP':
                    ftmat  = RT + "/" + self.extra_args.get("tmat")[i]
                    self.tmat[i] = np.load(ftmat, allow_pickle=True)
                    self.M[i] = ResMLP(input_dim  = len(self.extra_args.get('ord')[i]), 
                                       output_dim = len(self.tmat[i]), 
                                       int_dim    = self.extra_args.get('extrapar')[i]['INTDIM'], 
                                       N_layer    = self.extra_args.get('extrapar')[i]['NLAYER'])
                self.M[i] = self.M[i].to(self.device)
                self.M[i] = nn.DataParallel(self.M[i])
                self.M[i].load_state_dict(torch.load(fname, map_location=self.device))
                self.M[i] = self.M[i].module.to(self.device)
                self.M[i].eval()
                self.ord[i] = self.extra_args.get('ord')[i]

    def predict_cmb(self, model, X, einfo):
        X_mean = torch.Tensor(einfo.item()['X_mean']).to(self.device)
        X_std  = torch.Tensor(einfo.item()['X_std']).to(self.device)
        Y_mean = torch.Tensor(einfo.item()['Y_mean']).to(self.device)
        Y_std  = torch.Tensor(einfo.item()['Y_std']).to(self.device)
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            X_norm = (X - X_mean)/X_std
            X_norm = torch.nan_to_num(X_norm, nan=0)
            X_norm.to(self.device)
            M_pred = model(X_norm).to(self.device)
        return (M_pred.float()*Y_std.float() + Y_mean.float()).cpu().numpy()

    def predict_phi(self,model,X,einfo,tmat):
        X_mean=torch.Tensor(einfo.item()['X_mean']).to(self.device)
        X_std=torch.Tensor(einfo.item()['X_std']).to(self.device)
        Y_mean=einfo.item()['Y_mean']
        Y_std=einfo.item()['Y_std']
        Y_mean_2=torch.Tensor(einfo.item()['Y_mean_2']).to(self.device)
        Y_std_2=torch.Tensor(einfo.item()['Y_std_2']).to(self.device)
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            X_norm = (X - X_mean)/X_std
            X_norm=torch.nan_to_num(X_norm, nan=0)
            X_norm.to(self.device)
            M_pred = model(X_norm).to(self.device)
        y_pred = (M_pred.float()*Y_std_2.float() + Y_mean_2.float()).cpu().numpy()
        return np.exp(np.matmul(y_pred, tmat)*Y_std+Y_mean)
        
    def calculate(self, par):
        state = {}

        state.update({"ell": self.ell.astype(int)})
        cmb  = ['tt', 'te', 'ee', 'pp', 'bb']
        state.update({cmb[i]: np.zeros(self.lmax_theory) for i in range(5)})
        
        idx  = np.where(np.array(self.extra_args.get('eval'))[:3])[0]
        for i in idx:
            params = self.ord[i]
            p = np.array([par[key] for key in params])
            logAs  = p[params.index('logA')]
            tau    = p[params.index('tau')]
            norm    = np.exp(logAs)/np.exp(2*tau)
            lmax   = self.extra_args.get('extrapar')[i]['ellmax']
            state[cmb[i]][2:lmax] = self.predict_cmb(self.M[i], p, self.info[i])*norm
        state["et"] = state["te"]
        if self.extra_args.get('eval')[3]:
            state["pp"][2:len(phiphi)+2] = self.predict_phi(self.M[3], p, self.info[3], self.tmat[3])[0]

        return state

    def get_Cl(self, params, ell_factor=False, units="1", unit_included=True, Tcmb=2.7255):
        cls_old = self.calculate(par=params)
        cls_dict = {k : np.zeros(self.lmax_theory) for k in [ "tt", "te", "ee" , "et" , "bb" ]}
        cls_dict["ell"] = self.ell
        ls = self.ell
        
        if ell_factor:
            for k in [ "tt", "te", "ee" , "et" , "bb", "pp" ]:
                ls_fac = self.ell_factor(ls,k)
                cls_dict[k] = cls_old[k] * ls_fac
        else:
            for k in [ "tt", "te", "ee" , "et" , "bb", "pp" ]:
                cls_dict[k] = cls_old[k]
        if unit_included:
            unit=1
        else:
            for k in [ "tt", "te", "ee" , "et" , "bb", "pp" ]:
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