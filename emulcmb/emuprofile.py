import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cobaya.theories.emulcmbtrf.emulator import Supact, Affine, Better_Attention, Better_Transformer, ResBlock, TRF

class EmulProfile():
    
    def __init__(self, ROOTDIR, ttfilename, tefilename, eefilename, ttextraname, teextraname, eeextraname):
        intdim = 4
        nc = 16
        inttrf=5120
        device = 'cpu'
        
        self.model1 = TRF(input_dim=9,output_dim=4998,int_dim=intdim,int_trf=inttrf,N_channels=nc)
        self.model2 = TRF(input_dim=9,output_dim=4998,int_dim=intdim,int_trf=inttrf,N_channels=nc)
        self.model3 = TRF(input_dim=9,output_dim=4998,int_dim=intdim,int_trf=inttrf,N_channels=nc)

        self.model1 = self.model1.to(device)
        self.model2 = self.model2.to(device)
        self.model3 = self.model3.to(device)

        self.model1 = nn.DataParallel(self.model1)
        self.model2 = nn.DataParallel(self.model2)
        self.model3 = nn.DataParallel(self.model3)

        self.model1.load_state_dict(torch.load(ROOTDIR+"/"+ttfilename+'.pt',map_location=device))
        self.model2.load_state_dict(torch.load(ROOTDIR+"/"+tefilename+'.pt',map_location=device))
        self.model3.load_state_dict(torch.load(ROOTDIR+"/"+eefilename+'.pt',map_location=device))

        self.model1 = self.model1.module.to(device)
        self.model2 = self.model2.module.to(device)
        self.model3 = self.model3.module.to(device)

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()

        self.ell = np.arange(0,9052,1)
        self.lmax_theory = 9052

        self.extrainfo_TT = np.load(ROOTDIR+"/"+ttextraname,allow_pickle=True)
        self.extrainfo_TE = np.load(ROOTDIR+"/"+teextraname,allow_pickle=True)
        self.extrainfo_EE = np.load(ROOTDIR+"/"+eeextraname,allow_pickle=True)

    def predict(self,model,X, extrainfo):
        device = 'cpu'
        X_mean=torch.Tensor(extrainfo.item()['X_mean']).to(device)
        X_std=torch.Tensor(extrainfo.item()['X_std']).to(device)
        Y_mean=torch.Tensor(extrainfo.item()['Y_mean']).to(device)
        Y_std=torch.Tensor(extrainfo.item()['Y_std']).to(device)
        unused = np.array([0.06, -1, 0])
        X_send = np.concatenate((X, unused))
        X = torch.Tensor(X_send).to(device)
        with torch.no_grad():
            X_norm=((X - X_mean) / X_std)
            X_norm[:,6:]=0
            X_norm.to(device)
            pred=model(X_norm)
            M_pred=pred.to(device)
            y_pred = (M_pred.float() *Y_std.float()+Y_mean.float()).cpu().numpy()
        return y_pred
    
    def scaletrans(self,y_pred,X):
        for i in range(len(y_pred)):
            y_pred[i]=y_pred[i]*(np.exp(X[5]))/(np.exp(2*X[3]))
        return y_pred

    def get_Cl(self,cmb_params,ell_factor=False,units="1",unit_included=True,Tcmb=2.7255):
        #cmb_params need to be a numpy array in the order of [ombh2, omch2, H0, tau, ns, logAs]

        TT_rescale = self.predict(self.model1,cmb_params,self.extrainfo_TT)
        TE_rescale = self.predict(self.model2,cmb_params,self.extrainfo_TE)
        EE_rescale = self.predict(self.model3,cmb_params,self.extrainfo_EE)
        
        cldict={}
        cldict["ell"] = self.ell.astype(int)
        cldict["tt"]  = np.zeros(self.lmax_theory)
        cldict["te"]  = np.zeros(self.lmax_theory)
        cldict["bb"]  = np.zeros(self.lmax_theory)
        cldict["ee"]  = np.zeros(self.lmax_theory)
        cldict["tt"][2:5000] = self.scaletrans(TT_rescale,cmb_params)[0]
        cldict["te"][2:5000] = self.scaletrans(TE_rescale,cmb_params)[0]
        cldict["ee"][2:5000] = self.scaletrans(EE_rescale,cmb_params)[0]
        cldict["et"] = cldict["te"]

        if ell_factor:
            for k in [ "tt", "te", "ee" , "et" , "bb"]:
                ls_fac = self.ell_factor(ls,k)
                cldict[k] = cldict[k] * ls_fac
        else:
            for k in [ "tt", "te", "ee" , "et" , "bb"]:
                cldict[k] = cldict[k]
        if unit_included:
            unit=1
        else:
            for k in [ "tt", "te", "ee" , "et" , "bb" ]:
                unit = self.cmb_unit_factor(k, units, Tcmb)
                cldict[k] = cldict[k] * unit
        return cldict
        
    def ell_factor(self, ls: np.ndarray, spectra: str) -> np.ndarray:
        ellfac = np.ones_like(ls).astype(float)

        if spectra in ["tt", "te", "tb", "ee", "et", "eb", "bb", "bt", "be"]:
            ellfac = ls * (ls + 1.0) / (2.0 * np.pi)
        elif spectra in ["pt", "pe", "pb", "tp", "ep", "bp"]:
            ellfac = (ls * (ls + 1.0)) ** (3. / 2.) / (2.0 * np.pi)
        elif spectra in ["pp"]:
            ellfac = (ls * (ls + 1.0)) ** 2.0 / (2.0 * np.pi)
        return ellfac

    def cmb_unit_factor(self, spectra: str, units: str = "1", Tcmb: float = 2.7255) -> float:
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
