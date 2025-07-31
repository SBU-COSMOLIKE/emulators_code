from cobaya.likelihood import Likelihood
import numpy as np

class MyPkLikelihood(Likelihood):

    def initialize(self):
        self.kmax = 100.0
        self.linear_pk = None
        self.nonlinear_pk = None
        z1_mps = np.linspace(0,2,100,endpoint=False)
        z2_mps = np.linspace(2,10,10,endpoint=False)
        z3_mps = np.linspace(10,50,12)
        self.z_eval = np.concatenate((z1_mps,z2_mps,z3_mps),axis=0)
    def get_requirements(self):
        return {"omegabh2": None, "Pk_interpolator": {
        "z":self.z_eval,
        "k_max": 200,
        "nonlinear": (True,False),
        "vars_pairs": ([("delta_tot", "delta_tot")])
        }, "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
        'tt': 0
        },"omegach2":None, "H0": None, "ns": None, "As": None, "tau": None}

    def logp(self, **params):
        # Get linear P(k, z)

        self.linear_pk = self.provider.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=False, extrap_kmax = self.kmax)

        # Get non-linear P(k, z)
        self.nonlinear_pk = self.provider.get_Pk_interpolator(
                ("delta_tot", "delta_tot"), nonlinear=True, extrap_kmax = self.kmax
            )

        # Choose k range and evaluate both
        k = np.logspace(-4, np.log10(self.kmax), 2000)
        pk_lin = self.linear_pk.P(self.z_eval, k)
        pk_nonlin = self.nonlinear_pk.P(self.z_eval, k)



        # Return dummy log-likelihood
        return -0.5 