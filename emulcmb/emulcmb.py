import torch, os
import os.path as path
import torch.nn as nn
import numpy as np
from cobaya.theory import Theory
from cobaya.theories.emulcmb.emulator import ResBlock, ResMLP, TRF, CNNMLP
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
        self.ell = np.arange(0,self.lmax_theory,1)
        # TT, TE, EE, PHIPHI 
        imax = 4
        self.eval  = [False] * imax
        for name in ("M", "info", "ord", "tmat", "extrapar"):
            setattr(self, name, [None] * imax )
        for name in ("X_mean", "X_std", "Y_mean", "Y_std", "Y_mean_2", "Y_std_2"):
            setattr(self, name, [None] * imax)
        self.req   = [] 
        self.device = "cuda" if self.extra_args.get("device") == "cuda" and torch.cuda.is_available() else "cpu"

        if (teval := self.extra_args.get('eval')) is not None:
            for i in range(imax):
                self.eval[i] = (i<len(teval)) and bool(teval[i])
        self.eval = np.array(self.eval, dtype=bool)

        # BASIC CHECKS BEGINS ------------------------------------------------
        _required_lists = [
            ("extra", "Emulator CMB: Missing emulator file (extra) option"),
            ("ord", "Emulator CMB: Missing ord (parameter ordering) option"),
            ("file", "Emulator CMB: Missing emulator file option"),
            ("extrapar", "Emulator CMB: Missing extrapar option"),
        ]
        _mla_requirements = {
            "TRF":    ["ellmax","INTDIM","INTTRF","NCTRF"],
            "CNN":    ["ellmax","INTDIM","INTCNN"],
            "ResMLP": ["INTDIM","NLAYER","TMAT"],
        }
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
                raise ValueError('Emulator CMB: extrapar option not a dictionary')
            mla = self.extrapar[i].get('MLA')
            if mla is None or (isinstance(mla, str) and mla.strip().lower() == "none"):
                raise ValueError(f'Emulator CMB: Missing extrapar MLA option')
            try:
                req_keys = _mla_requirements[mla]
            except KeyError:
                raise KeyError(f"Emulator CMB: Unknown MLA option: {mla}")
            miss = [k for k in req_keys if k not in self.extrapar[i]]
            if miss:
                raise KeyError(f"Emulator CMB: Missing extrapar keys for {mla}: {miss}")
        # BASIC CHECKS ENDS ------------------------------------------------
        
        for i in range(imax):
            if not self.eval[i]:
                continue
    
            self.info[i] = path.join(RT,self.extra_args['extra'][i])
            self.info[i] = np.load(self.info[i],allow_pickle=True)
            self.X_mean[i] = torch.Tensor(self.info[i].item()['X_mean']).to(self.device)
            self.X_std[i] = torch.Tensor(self.info[i].item()['X_std']).to(self.device)
            if i == 3:
                self.Y_mean_2[i] = self.info[i].item()['Y_mean']
                self.Y_std_2[i]  = self.info[i].item()['Y_std']
                self.Y_mean[i] = torch.Tensor(self.info[i].item()['Y_mean_2']).to(self.device)
                self.Y_std[i] = torch.Tensor(self.info[i].item()['Y_std_2']).to(self.device)
            else:
                self.Y_mean[i] = torch.Tensor(self.info[i].item()['Y_mean']).to(self.device)
                self.Y_std[i]  = torch.Tensor(self.info[i].item()['Y_std']).to(self.device)

            self.ord[i] = self.extra_args['ord'][i]           
            
            if self.extrapar[i]['MLA'] == 'TRF':
                self.M[i] = TRF(input_dim = len(self.ord[i]),
                                output_dim = self.extrapar[i]['ellmax']-2,
                                int_dim = self.extrapar[i]['INTDIM'],
                                int_trf = self.extrapar[i]['INTTRF'],
                                N_channels = self.extrapar[i]['NCTRF'])
            elif self.extrapar[i]['MLA'] == 'CNN':
                self.M[i] = CNNMLP(input_dim = len(self.ord[i]),
                                   output_dim = self.extrapar[i]['ellmax']-2,
                                   int_dim = self.extrapar[i]['INTDIM'],
                                   cnn_dim = self.extrapar[i]['INTCNN'])
            elif self.extrapar[i]['MLA'] == 'ResMLP':
                self.tmat[i] = np.load(path.join(RT,self.extrapar[i]["TMAT"]),allow_pickle=True)
                self.M[i] = ResMLP(input_dim = len(self.ord[i]), 
                                   output_dim = len(self.tmat[i]), 
                                   int_dim = self.extrapar[i]['INTDIM'], 
                                   N_layer = self.extrapar[i]['NLAYER'])
            self.M[i] = self.M[i].to(self.device)
            self.M[i] = nn.DataParallel(self.M[i])
            self.M[i].load_state_dict(torch.load(path.join(RT,self.extra_args["file"][i]),map_location=self.device))
            self.M[i] = self.M[i].module.to(self.device)
            self.M[i].eval()
            self.req.extend(self.ord[i])
        self.req = list(set(self.req))
        d = {}
        for i in self.req:
            d[i] = None
        self.req = d

        self.exponent_map = {
            **dict.fromkeys(["tt","te","tb","ee","et","eb","bb","bt","be"], 1.0),
            **dict.fromkeys(["pt","pe","pb","tp","ep","bp"], 1.5),
            "pp": 2.0,
        }
        self.cmb  = ['tt', 'te', 'ee', 'pp', 'bb']

    def get_requirements(self):
        return self.req

    def predict_data_vector(self, X, i):
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            X_norm = torch.nan_to_num((X-self.X_mean[i])/self.X_std[i],nan=0).to(self.device)
            M_pred = self.M[i](X_norm).to(self.device)
        y_pred = (M_pred*self.Y_std[i] + self.Y_mean[i]).cpu().detach().numpy()
        if i != 3:
            return y_pred
        else:
            return np.exp(np.matmul(y_pred,self.tmat[i])*self.Y_std_2[i] + self.Y_mean_2[i])     
        
    def calculate(self, state, want_derived=False, **par):
        state.update({self.cmb[i]: np.zeros(self.lmax_theory) for i in range(len(self.cmb))} |
                     {"ell": self.ell.astype(int)})
        idx  = np.where(self.eval[:3])[0]
        for i in idx:
            params = self.ord[i]
            X = np.array([par[key] for key in params])
            logAs  = X[params.index('logA')]
            tau    = X[params.index('tau')]
            norm   = np.exp(logAs)/np.exp(2*tau)
            lmax   = self.extra_args.get('extrapar')[i]['ellmax']
            state[self.cmb[i]][2:lmax] = self.predict_data_vector(X,i)*norm
        state["et"] = state["te"]
        i=3
        if self.eval[i]:
            params = self.ord[i]
            X = np.array([par[key] for key in params])
            phiphi = self.predict_data_vector(X,i)[0]
            state["pp"][2:len(phiphi)+2] = phiphi
        np.save('cmbemultest.npy',state)
        return True

    def get_Cl(self, ell_factor = False, units = "1", unit_included = True, Tcmb=2.7255):
        return { k: self.current_state[k]
                    * (self.ell_factor(self.ell, k) if ell_factor else 1)
                    * (1 if unit_included else self.cmb_unit_factor(k, units, Tcmb))
                    for k in [ "tt","te","ee","et","bb","pp" ] } | {"ell": self.ell}
        
    def ell_factor(self, ls: np.ndarray, spectra: str) -> np.ndarray:
        exp = self.exponent_map.get(spectra)
        return ((ls*(ls+1))**exp/(2*np.pi)) if exp is not None else np.ones_like(ls, float)

    def cmb_unit_factor(self, spectra: str,
                        units: str = "1",
                        Tcmb: float = 2.7255) -> float:
        u = self._cmb_unit_factor(units, Tcmb)
        p = 1.0 / math.sqrt(2.0 * math.pi)
        return math.prod(u if c in ("t","e","b") else p if c == "p" else 1.0 for c in spectra.lower())
