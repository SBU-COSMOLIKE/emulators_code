import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, Matern, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
import joblib


d=np.load('something',allow_pickle=True)
dv=np.load('something',allow_pickle=True)

extrainfo=np.load("something",allow_pickle=True)
X_mean=extrainfo.item()['X_mean']
X_std=extrainfo.item()['X_std']
Y_mean=extrainfo.item()['Y_mean']
Y_std=extrainfo.item()['Y_std']

d=(d-X_mean)/X_std

dv=(dv-Y_mean)/Y_std
#defining the model. Here we have an RBF kernel+white kernel. You can change it
#according to your tweaking.
kernel = C(1.0, (1e-5, 1e5)) * RBF(length_scale=1)+ WhiteKernel(noise_level=0.5)
def optimizer(obj_func, x0, bounds):
    res = scipy.optimize.minimize(
        obj_func, x0, bounds=bounds, method="L-BFGS-B", jac=True,
        options={'maxiter':1000})
    return res.x, res.fun


gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,optimizer=optimizer)

X_train, X_test, y_train, y_test = train_test_split(d,dv, test_size=0.3, random_state=0)

gp.fit(X_train, y_train)
gp.optimizer=None #need to dump the optimizer before saving

joblib.dump(gp, 'directory.joblib')
