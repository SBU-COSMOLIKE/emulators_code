import numpy as np
import os, joblib

class emultheta():    
    def __init__(self, extra_args):
        self.extra_args = extra_args
        RT = os.environ.get("ROOTDIR")
        self.M      = [None]
        self.info   = [None]
        self.ord    = [None]
        self.device = 'cpu'

        fname  = RT + "/" + self.extra_args.get("file")[0]
        fextra = RT + "/" + self.extra_args.get("extra")[0]
        self.info[0] = np.load(fextra,allow_pickle=True)
        self.MLA = self.extra_args.get('extrapar')[0]['MLA']
        self.ord[0]  = self.extra_args.get('ord')[0]
        if self.MLA == "GP":
            self.M[0]    = joblib.load(fname)
        elif self.MLA == "simpMLP":
            intdim = self.extra_args.get('extrapar')[0]['INTDIM']
            Nlayer = self.extra_args.get('extrapar')[0]['NLAYER']
            self.M[0] = simpMLP(input_dim=len(self.ord[0]),output_dim=1,int_dim=intdim,N_layer=Nlayer)
            self.M[0] = self.M[0].to(self.device)
            self.M[0] = nn.DataParallel(self.M[0])
            self.M[0].load_state_dict(torch.load(fname, map_location=self.device))
            self.M[0] = self.M[0].module.to(self.device)
            self.M[0].eval()

    def predict(self, X, Y_mean, Y_std, model):
        with torch.no_grad():
            X = torch.Tensor(X).to(self.device)
            
            pred = model(X)
            
            M_pred = pred.to(self.device).float().cpu().numpy()
            y_pred = (M_pred *Y_std+Y_mean)
            
        return y_pred
        

    def calculate(self, par):


        state = {}
        X_mean = self.info[0].item()['X_mean']
        Y_mean = self.info[0].item()['Y_mean']
        X_std  = self.info[0].item()['X_std']
        Y_std  = self.info[0].item()['Y_std']
        p =  np.array([par[key] for key in self.ord[0]]) - X_mean
        if self.MLA == "GP":
            H0 = self.M[0].predict(p/X_std)[0]*Y_std[0] + Y_mean[0]
        elif self.MLA == "simpMLP":
            p0 = p/X_std
            H0 = self.predict(p0, Y_mean, Y_std, self.M[0])[0][0]
        
        h2       = (H0/100.0)**2
        omegamh2 = par["omegamh2"]

        state.update({"H0": par["H0"]})
        state.update({"omegam": omegamh2/h2})    
   
        return state

    def get_H0(self, params):
        state = self.calculate(params)
        return state["H0"]

    def get_omegam(self, params):
        state = self.calculate(params)
        return state["omegam"]