import torch
import torch.nn as nn
import numpy as np
import os
from cobaya.theories.emulbaosn.emulator import ResBlock, ResMLP, TRF
from scipy import interpolate

class emulbaosn():    
    def __init__(self, extra_args):
        self.extra_args = extra_args
        RT = os.environ.get("ROOTDIR")
        self.M      = [None, None] # dl, H(z)
        self.info   = [None, None] # dl, H(z)
        self.z      = [None, None] # dl, H(z)
        self.tmat   = [None, None] # dl, H(z)
        self.offset = [None, None] # dl, H(z)
        self.ord    = [None, None] # dl, H(z)
        self.device = self.extra_args.get("device")
        self.zstep = np.arange(0,3,0.0048)
        self.dz = (self.zstep[1] - self.zstep[0])  # assuming uniform spacing
        
        self.device = "cuda" if self.extra_args.get("device") == "cuda" and torch.cuda.is_available() else "cpu"

        if (teval := self.extra_args.get('eval')) is not None:
            for i in range(imax):
                self.eval[i] = (i<len(teval)) and bool(teval[i])
        self.eval = np.array(self.eval, dtype=bool)

        # H(z) ------------------------------------------------------------
        i = 1
        if self.extra_args.get('eval')[i]:
            fzlin  = RT + "/" + self.extra_args.get("zlin")[i]
            self.z[i]    = np.load(fzlin, allow_pickle=True)
            
            fname  = RT + "/" + self.extra_args.get("file")[i]
            fextra = RT + "/" + self.extra_args.get("extra")[i]
            ftmat  = RT + "/" + self.extra_args.get("tmat")[i]
            self.info[i] = np.load(fextra, allow_pickle=True)
            self.tmat[i] = np.load(ftmat, allow_pickle=True)
            self.offset[i] = self.extra_args.get('extrapar')[i]['offset']
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
            self.req.extend(self.ord[i])
        # SN ------------------------------------------------------------
        i = 0   
        if self.extra_args.get('eval')[i]:
            if self.extra_args.get("method")[i] == "NN":
                fzlin  = RT + "/" + self.extra_args.get("zlin")[i]
                self.z[i]    = np.load(fzlin, allow_pickle=True)
                
                fname  = RT + "/" + self.extra_args.get("file")[i]
                fextra = RT + "/" + self.extra_args.get("extra")[i]
                ftmat  = RT + "/" + self.extra_args.get("tmat")[i]
                self.info[i] = np.load(fextra, allow_pickle=True)
                self.tmat[i] = np.load(ftmat, allow_pickle=True)
                self.offset[i] = self.extra_args.get('extrapar')[i]['offset']
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
                self.req.extend(self.ord[i])
            elif self.extra_args.get("method")[i] == "INT":
                if not self.extra_args.get('eval')[0]:
                    raise ValueError('wrong flag - cant integrate H(z)')
                zmin = self.extra_args.get('extrapar')[i]['ZMIN']
                zmax = self.extra_args.get('extrapar')[i]['ZMAX']
                if (zmax > self.z[1][-1]):
                    zmax = self.z[1][-1]
                NZ   = self.extra_args.get('extrapar')[i]['NZ']
                self.z[i] = np.linspace(zmin, zmax, NZ)
                self.ord[i] = self.extra_args.get('ord')[1] # Same as H(z)
                self.req.extend(self.ord[i])

    def predict(self, model, X, extrainfo, transform_matrix, offset):
        X_mean   = torch.Tensor(extrainfo.item()['X_mean']).to(self.device)
        X_std    = torch.Tensor(extrainfo.item()['X_std']).to(self.device)
        Y_mean   = extrainfo.item()['Y_mean']
        Y_std    = extrainfo.item()['Y_std']
        Y_mean_2 = torch.Tensor(extrainfo.item()['Y_mean_2']).to(self.device)
        Y_std_2  = torch.Tensor(extrainfo.item()['Y_std_2']).to(self.device)
        X = torch.Tensor(X).to(self.device)
        with torch.no_grad():
            X_norm = ((X - X_mean) / X_std)
            X_norm.to(self.device)
            pred = model(X_norm)
            M_pred = pred.to(self.device)
            y_pred = (M_pred.float() *Y_std_2.float()+Y_mean_2.float()).cpu().numpy()
            y_pred = np.matmul(y_pred,transform_matrix)*Y_std+Y_mean
            y_pred = np.exp(y_pred) - offset
        return y_pred[0]

    def HtoDl(self, H):
        c = 2.99792458e5
        func = interpolate.interp1d(self.z[1],c/H,fill_value="extrapolate")
        integrand = func(self.zstep)
        dl = np.zeros_like(integrand)
        f0 = integrand[0:-2:2]
        f1 = integrand[1:-1:2]
        f2 = integrand[2::2]
        simpson_chunks = self.dz / 3 * (f0 + 4 * f1 + f2)
        dl[2::2] = np.cumsum(simpson_chunks)
        dl[1::2] = (dl[0:-2:2] + dl[2::2]) / 2
        dl*=(1+self.zstep)
        return interpolate.interp1d(self.zstep,dl,fill_value="extrapolate")

    def calculate(self, par):       
        state = {}
        out    = ["dl","H"]
        idx    = np.where(np.array(self.extra_args.get('eval'))[:2])[0]
        # H(z) ------------------------------------------------------------
        i = 1
        if i in idx:
            params = self.extra_args.get('ord')[i]
            p = np.array([par[key] for key in params])
            state[out[i]] = self.predict(self.M[i], p, self.info[i], self.tmat[i], self.offset[i])
        # SN ------------------------------------------------------------
        i = 0
        if i in idx:
            if self.extra_args.get("method")[i] == "NN":
                params = self.extra_args.get('ord')[i]
                p = np.array([par[key] for key in params])
                state[out[i]] = self.predict(self.M[i], p, self.info[i], self.tmat[i], self.offset[i])
            elif self.extra_args.get("method")[i] == "INT":
                state[out[0]] = self.HtoDl(state["H"])(self.z[0])
        return state


    def get_angular_diameter_distance(self, z):
        d_l = self.current_state["dl"].copy()
        d_a = d_l/(1.0 + self.z[0])**2
        D_A_interpolate = interpolate.interp1d(self.z[0], d_a)
        D_A = D_A_interpolate(z)
        try:
            l = len(D_A)
        except:
            D_A = np.array([D_A])
        else:
            l = 1
        return D_A

    def get_angular_diameter_distance_2(self, zpair):
        z_1, z_2 = zpair[0]
        
        if z_1 >= z_2:
            return 0
        else:
            da1 = self.get_angular_diameter_distance(z_1)
            da2 = self.get_angular_diameter_distance(z_2)
            cd1 = da1*(1+z_1)
            cd2 = da2*(1+z_2)
            return (cd2-cd1)/(1+z_2)

    def get_Hubble(self, z, units = "km/s/Mpc"):
        H = self.current_state["H"].copy()
        H_interpolate = interpolate.interp1d(self.z[1], H)
        H_arr = H_interpolate(z)
        try:
            l = len(H_arr)
        except:
            H_arr = np.array([H_arr])
        else:
            l = 1
        if units == "1/Mpc":
            H_arr /= 2.99792458e5
        return H_arr