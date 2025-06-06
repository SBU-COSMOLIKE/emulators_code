import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cobaya.theories.cosmo import BoltzmannBase
from cobaya.theories.emulsn import emulsn
from cobaya.typing import InfoDict
from cobaya.theories.emulbaosn.emulator import Supact, Affine, Better_Attention, Better_Transformer, ResBlock, ResMLP, TRF
import joblib
import scipy
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class emulbaosn(emulsn):
    

    extra_args: InfoDict = { }
    

    def initialize(self):
        super().initialize()
        self.ordering = self.extra_args.get('ordering')
        self.thetaordering = 0.0
        self.rdragordering = self.extra_args.get('rdragordering')
        self.z_lin_dl = self.extra_args.get('zlindl')
        self.z_lin_H = self.extra_args.get('zlinH')
        self.extradllevel = self.extra_args.get('extradllevel')
        
        
        self.ROOT = os.environ.get("ROOTDIR")


        self.PATH4 = self.ROOT + "/" + self.extra_args.get('dlfilename')
        self.PATH5 = self.ROOT + "/" + self.extra_args.get('Hfilename')
        self.extrainfo_dl = np.load(self.ROOT+"/"+self.extra_args.get('dlextraname'), allow_pickle=True)
        self.transmat_dl  = np.load(self.ROOT+"/"+self.extra_args.get('dltransmat'), allow_pickle=True)
        self.extrainfo_H = np.load(self.ROOT+"/"+self.extra_args.get('Hextraname'), allow_pickle=True)
        self.transmat_H  = np.load(self.ROOT+"/"+self.extra_args.get('Htransmat'), allow_pickle=True)


        
        self.PATH7 = "INIT"  # File that contains GP model for theta to H0

        
        self.extrainfo_GP = 0.0 # extra info file for GP of theta to H0
        
        intdim = 4
        nlayer = 4
        intdim_simple = 1
        nlayer_simple = 1

        device = 'cpu'


        self.model4 = ResMLP(input_dim=len(self.ordering), output_dim=len(self.transmat_dl), int_dim=intdim, N_layer=nlayer)
        self.model5 = ResMLP(input_dim=len(self.ordering), output_dim=len(self.transmat_H), int_dim=intdim_simple, N_layer=nlayer_simple)

        self.model7 = 0.0 # load GP model for theta to H0

        self.model4 = self.model4.to(device)
        self.model5 = self.model5.to(device)


        self.model4 = nn.DataParallel(self.model4)
        self.model5 = nn.DataParallel(self.model5)


        self.model4.load_state_dict(torch.load(self.PATH4+'.pt',map_location=device))
        self.model4 = self.model4.module.to(device)
        self.model4.eval()
        self.model5.load_state_dict(torch.load(self.PATH5+'.pt',map_location=device))
        self.model5 = self.model5.module.to(device)
        self.model5.eval()

        self.PATH6 = self.ROOT + "/" + self.extra_args.get('rdragfilename')

        # load GP model for theta to H0
        self.model6 = joblib.load(self.PATH6) 

        self.extrainfo_rdrag = np.load(self.ROOT + 
                                    "/"  + 
                                    self.extra_args.get('rdragextraname'), 
                                                        allow_pickle=True)

        self.testh0 = -1



    def predict_H(self,model,X, extrainfo,transform_matrix):
        device = 'cpu'
        X_mean=torch.Tensor(extrainfo.item()['X_mean']).to(device)
        X_std=torch.Tensor(extrainfo.item()['X_std']).to(device)
        Y_mean=extrainfo.item()['Y_mean']
        Y_std=extrainfo.item()['Y_std']
        Y_mean_2=torch.Tensor(extrainfo.item()['Y_mean_2']).to(device)
        Y_std_2=torch.Tensor(extrainfo.item()['Y_std_2']).to(device)
        

        X = torch.Tensor(X).to(device)

        with torch.no_grad():
            X_norm=((X - X_mean) / X_std)



            X_norm.to(device)

            
            pred=model(X_norm)
            
            
            M_pred=pred.to(device)
            y_pred = (M_pred.float() *Y_std_2.float()+Y_mean_2.float()).cpu().numpy()
            y_pred = np.matmul(y_pred,transform_matrix)*Y_std+Y_mean
            y_pred = np.exp(y_pred)
        return y_pred[0]


        
    def calculate(self, state, want_derived=True, **params):
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
        cmb_param["omm"] = (cmb_param["omegabh2"]+cmb_param["omegach2"])/(cmb_param["H0"]/100)**2

        cmb_params = np.array([cmb_param[key] for key in self.ordering])

        state["dl"] = self.predict_dl(self.model4, cmb_params, self.extrainfo_dl, self.transmat_dl)
        state["H"] = self.predict_H(self.model5, cmb_params, self.extrainfo_H, self.transmat_H)

        vd = np.array([cmb_param[key] for key in self.rdragordering]) - self.extrainfo_rdrag.item()['X_mean']
        rdrag = self.model6.predict(vd/self.extrainfo_rdrag.item()['X_std'])[0]*self.extrainfo_rdrag.item()['Y_std'][0] + self.extrainfo_rdrag.item()['Y_mean'][0]
        if "derived" in state:
            state["derived"].update({"rdrag": rdrag})#test
        elif "rdrag" not in state["derived"]:
            state["derived"]={"rdrag": rdrag}
        else:
            state["derived"]["rdrag"]=rdrag
        return True

    def get_rdrag(self):
        return self.model6.predict(vd/self.extrainfo_rdrag.item()['X_std'])[0]*self.extrainfo_rdrag.item()['Y_std'][0] + self.extrainfo_rdrag.item()['Y_mean'][0]
    
    def get_angular_diameter_distance(self,z):
        d_l = self.current_state["dl"].copy()

        z_lin = np.load(self.z_lin_dl, allow_pickle=True)

        d_a = d_l/(1+z_lin)**2


        D_A_interpolate = interpolate.interp1d(z_lin, d_a)

        
        D_A = D_A_interpolate(z)

        try:
            l = len(D_A)
        except:
            D_A = np.array([D_A])
        else:
            l = 1


        return D_A

    def get_angular_diameter_distance_2(self,zpair):

        z_1, z_2 = zpair[0]
        
        if z_1 >= z_2:
            return 0
        else:
            da1 = self.get_angular_diameter_distance(z_1)
            da2 = self.get_angular_diameter_distance(z_2)

            cd1 = da1*(1+z_1)
            cd2 = da2*(1+z_2)
            
            return (cd2-cd1)/(1+z_2)


    def get_Hubble(self,z,units="km/s/Mpc"):
        H = self.current_state["H"].copy()

        z_lin = np.load(self.z_lin_H, allow_pickle=True)

        H_interpolate = interpolate.interp1d(z_lin, H)

        H_arr = H_interpolate(z)
        try:
            l = len(H_arr)
        except:
            H_arr = np.array([H_arr])
        else:
            l = 1

        if units=="1/Mpc":
            H_arr/=2.99792458e5

        return H_arr

    def get_can_support_params(self):
        return [ "omega_b", "omega_cdm", "h" ]
    