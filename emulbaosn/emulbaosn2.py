<<<<<<< HEAD
import torch
import torch.nn as nn
import numpy as np
import os
import os.path as path
from cobaya.theories.emulbaosn.emulator import ResBlock, ResMLP, TRF
from scipy import interpolate

class emulbaosn():    
    def __init__(self, extra_args):
        self.extra_args = extra_args
        RT = os.environ.get("ROOTDIR")
        self.imax = 2
        for name in ("M", "info", "ord", "z", "tmat", "extrapar", "offset"):
            setattr(self, name, [None] * self.imax)
        self.req = [] 
        self.device = "cuda" if self.extra_args.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
                
        # BASIC CHECKS BEGINS --------------------------------------------------
        _required_lists = [
            ("extra", "Emulator BAOSN: Missing emulator file (extra) option"),
            ("ord", "Emulator BAOSN: Missing ord (parameter ordering) option"),
            ("file", "Emulator BAOSN: Missing emulator file option"),
            ("extrapar", "Emulator BAOSN: Missing extrapar option"),
        ]
        for key, msg in _required_lists:
            if (tmp := self.extra_args.get(key)) is None or (len(tmp)<self.imax):
                raise ValueError(msg)        
            # H(Z) -------------------------------------------------------------
            i = 1
            if tmp[i] is None or (isinstance(tmp[i], str) and tmp[i].strip().lower()=="none"):
                raise ValueError(msg)            
        _mla_requirements = {
            "ResMLP": ["INTDIM","NLAYER","TMAT",'ZLIN','offset'],
            "INT": ["ZMIN","ZMAX","NZ"],
        }
        for i in range(self.imax):
            self.extrapar[i] = self.extra_args["extrapar"][i].copy()
            if not isinstance(self.extrapar[i], dict):
                raise ValueError('Emulator BAOSN: extrapar option not a dictionary')
            mla = self.extrapar[i].get('MLA')
            if mla is None or (isinstance(mla, str) and mla.strip().lower() == "none"):
                raise ValueError(f'Emulator BAOSN: Missing extrapar MLA option')
            try:
                req_keys = _mla_requirements[mla]
            except KeyError:
                raise KeyError(f"Emulator BAOSN: Unknown MLA option: {mla}")
            miss = [k for k in req_keys if k not in self.extrapar[i]]
            if miss:
                raise KeyError(f"Emulator BAOSN: Missing extrapar keys for {mla}: {miss}")
        # BASIC CHECKS ENDS ----------------------------------------------------

        # H(Z) -----------------------------------------------------------------
        i = 1
        self.info[i] = np.load(path.join(RT,self.extra_args.get("extra")[i]),allow_pickle=True)
        self.z[i]    = np.load(path.join(RT,self.extrapar[i]['ZLIN']),allow_pickle=True)
        self.tmat[i] = np.load(path.join(RT,self.extrapar[i]["TMAT"]),allow_pickle=True)

        self.ord[i] = self.extra_args['ord'][i]  
        self.offset[i] = self.extrapar[i]['offset']
        
        if self.extrapar[i]['MLA'].strip().lower() == 'resmlp':
            self.M[i] = ResMLP(input_dim = len(self.ord[i]), 
                               output_dim = len(self.tmat[i]), 
                               int_dim = self.extrapar[i]['INTDIM'], 
                               N_layer = self.extrapar[i]['NLAYER'])
        self.M[i] = self.M[i].to(self.device)
        self.M[i] = nn.DataParallel(self.M[i])
        self.M[i].load_state_dict(torch.load(path.join(RT,self.extra_args.get("file")[i]),map_location=self.device))
        self.M[i] = self.M[i].module.to(self.device)
        self.M[i].eval()
        self.req.extend(self.ord[i])
        # SN(Z) ----------------------------------------------------------------
        i = 0
        if self.extrapar[i]['MLA'].strip().lower() == 'int':
            zmin = self.extrapar[i]['ZMIN']
            zmax = self.extrapar[i]['ZMAX']
            if (zmax > self.z[1][-1]):
                zmax = self.z[1][-1]
            NZ = self.extrapar[i]['NZ']
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

    def cumulative_simpson(self, z, y):
        """
        Compute the cumulative integral of y(z) over z using composite Simpson’s rule
        on a uniform grid. Returns an array C of the same length as z, where
        C[k] = ∫_z[0]^z[k] y(z') dz'.
        """
        n = len(z)
        if n < 3 or (n - 1) % 2 != 0:
            raise ValueError("Need an odd number of points (even number of intervals).")
        dz = z[1] - z[0]
        
        # Simpson contributions on each pair of intervals [z[2m], z[2m+2]]
        # there are (n-1)/2 such chunks
        f0 = y[:-2:2]    # y[0], y[2], y[4], …
        f1 = y[1:-1:2]   # y[1], y[3], y[5], …
        f2 = y[2::2]     # y[2], y[4], y[6], …
        chunks = dz/3 * (f0 + 4*f1 + f2)
        
        # cumulative sum of the full‐chunk integrals at even indices
        cum_even = np.concatenate(([0.0], np.cumsum(chunks)))
        
        # build the full cumulative array
        C = np.empty_like(y)
        C[0] = 0.0
        C[2::2] = cum_even[1:]               # at z[2], z[4], …
        C[1::2] = (cum_even[:-1] + cum_even[1:]) / 2
        return C

    def calculate(self, par):       
        state = {}
        out = ["dl","H"]
        # z --------------------------------------------------------------------
        state['z'] = self.z[0]
        # H(z) -----------------------------------------------------------------
        i = 1
        params = self.extra_args.get('ord')[i]
        p = np.array([par[key] for key in params])
        tmp = self.predict(self.M[i], p, self.info[i], self.tmat[i], self.offset[i])

        # forcing H(z) and dL(z) to be defined on the same z-grid
        state[out[i]] = interpolate.interp1d(self.z[1], tmp, 
                                             kind='cubic',
                                             assume_sorted=True,
                                             fill_value="extrapolate")(self.z[0])
        # SN -------------------------------------------------------------------
        i = 0
        func  = interpolate.interp1d(self.z[1], 2.99792458e5/state["H"],
                                     kind='cubic',
                                     assume_sorted=True,
                                     fill_value="extrapolate")
        zstep = np.linspace(0.0, self.z[0][-1], 2*len(self.z[0])+1)
        dl    = self.cumulative_simpson(zstep, func(zstep))*(1 + zstep)
        state[out[i]] = interpolate.interp1d(zstep, dl, 
                                             kind='cubic',
                                             assume_sorted=True,
                                             fill_value="extrapolate")(self.z[0])
        return state
=======
import torch
import torch.nn as nn
import numpy as np
import os
from cobaya.theories.emulbaosn.emulator import ResBlock, ResMLP
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
>>>>>>> 1d6866f (update phiphi)
