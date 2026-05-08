"""
Cobaya theory that provides the non-linear correction to 
the matter perturbation using the EFT of large scale structure.
NO MASSIVE NEUTRINOS in the integrals.
"""

"""-----------------------------------------------------------------------------
--------------------------------------------------------------------------------
-----------------------------------------------------------------------------"""

import os, sys
import numpy as np
from cobaya.theory import Theory
from typing import Mapping, Iterable
from cobaya.typing import empty_dict, InfoDict
from scipy.interpolate import interp1d
from .eft_emulator import unified_emulator#, unified_model, loop_emulator
import scipy
import torch

reference_nonlin_list = ['none']
reference_nonlin_exclude = []
try:
    import euclidemu2 as ee2
    reference_nonlin_list.append('euclidemu2')
except:
    reference_nonlin_exclude.append('euclidemu2')

print('[eft] Available reference spectra:')
print('\t', reference_nonlin_list)
print('[eft] Unavailable reference spectra:')
print('\t', reference_nonlin_exclude)

pi2=2*np.pi # convinient constant that makes the code more clear

"""-----------------------------------------------------------------------------
--------------------------------------------------------------------------------
-----------------------------------------------------------------------------"""

# timing for debug
from contextlib import contextmanager
import time
@contextmanager
def timer(label):
  t0 = time.perf_counter()
  yield
  print(f"{label}: {time.perf_counter() - t0:.4f}s")


"""-----------------------------------------------------------------------------
--------------------------------------------------------------------------------
-----------------------------------------------------------------------------"""

class emul_eftoflss(Theory):
    renames: Mapping[str, str] = empty_dict
    #extra_args: InfoDict = { }
    _must_provide: dict
    path: str

    def initialize(self):
        super().initialize()
        ROOTDIR = os.environ.get("ROOTDIR")

        #----------------------------------------------------------
        # EFT INFO 
        # EFT is computed with 30 k per decade for nearly 7 decades
        # Cutoff of the integrals is Lambda=50

        k = np.logspace(-5, 2, 210)
        kmask = k<50

        self.k = k[kmask]
        self.logk = np.log(self.k)
        self.len_k = len(self.k)

        self.k2 = self.k * self.k 
        self.k4 = self.k2*self.k2

        # load emulator
        self.unified_emulator = unified_emulator(ROOTDIR+\
            '/external_modules/data/emultrf/emul_eftoflss/twoloop_w0wa_unified')

        self.k_uv = self.extra_args['k_uv']

        # use a reference pnl to fit the counterterms
        self.reference_nonlin = self.extra_args['reference_nonlin']
        assert self.reference_nonlin in reference_nonlin_list

        if 'none' != self.reference_nonlin:
            self.k_fit = self.extra_args['k_fit']
            if self.reference_nonlin == 'euclidemu2':
                print("[eft] Using Euclid Emulator 2 to calibrate EFT counterterms")
                self.external_boost_fcn = self.compute_ee2_boost
        else:
            print("[eft] Not calibrating counterterms to an external model.",
                "Values are set entirely by Spline")



        # counterterms PCA if specified
        eft_pca_file = self.extra_args['eft_pca_file']
        self.pca_evecs = np.diag(np.ones(45))
        self.nparams = 45
        if 'none' != eft_pca_file:
            print('[eft] Loading EFT PCA:',pca_file)
            self.pca_evecs = np.loadtxt(ROOT_DIR+\
                '/external_modules/data/emultrf/emul_eft/'+pca_file)
            if len(self.pca_evecs.shape) == 1:
                # special case where there was only one PC. 
                self.pca_evecs = np.atleast_2d(self.pca_evecs).T
            self.nparams = self.pca_evecs.shape[1]
            print('[eft] Can use up to {} EFT PC amplitudes.'.format(self.nparams))
        else:
            print("[eft] Not using any PCA for EFT parameters")
        

        #------------------------------------------
        # Accuracy boost. Sets density of redshifts

        self.accuracyboost = self.extra_args['accuracyboost']

        tmp=int(min(120 + 20*self.accuracyboost,250))
        self.z = np.concatenate((np.linspace(0,3.0,max(50,int(0.75*tmp))), 
            np.linspace(3.01,9.99,max(10,int(0.25*tmp)))),axis=0)
        self.a = 1/(1+self.z)
        self.len_z = len(self.z)

        #----------------------------------------
        # Lambda functions for derived quantities
        self.calc_omegam = lambda omegabh2, omegach2, h, mnu: \
            (omegabh2 + omegach2 + (mnu*(3.046/3)**0.75)/94.0708)/(h*h)
        self.calc_omegab = lambda omegabh2, h: omegabh2/(h*h)

        #--------------------------
        # Precomputing spline basis
        self.t = np.array([0,0,0,0,1,2,3,4,5,6,6,6,6])/6 # knot vector
        self.zmask = self.z<=4 # The z which we spline
        x = np.linspace(0,1,np.sum(self.zmask))
        basis = scipy.interpolate.BSpline.design_matrix(x, self.t, 3)
        self.c_basis = basis.toarray().T 

    """----------------------------------------------------------------------"""
    # COBAYA

    def get_requirements(self):
        return {
            "omegabh2": None,
            "omegach2": None,
            "As": None,
            "ns": None,
            "mnu": None, # Needed to convert to omegam
            "H0": None,
            "w": None,
            "wa": None
        }

    def get_can_provide(self):
        return {'non_linear_ratio': None}

    """----------------------------------------------------------------------"""
    # External reference non-linear boosts

    def compute_ee2_boost(self):
        omegabh2 = self.provider.get_param('omegabh2')
        omegach2 = self.provider.get_param('omegach2')
        h = self.provider.get_param('H0')/100
        mnu = self.provider.get_param('mnu')

        p = {
            'Omm'  : self.calc_omegam(omegabh2, omegach2, h, mnu),
            'As'   : self.provider.get_param("As"),
            'Omb'  : self.calc_omegab(omegabh2, h),
            'ns'   : self.provider.get_param("ns"),
            'h'    : h,
            'mnu'  : mnu, 
            'w'    : self.provider.get_param("w"),
            'wa'   : self.provider.get_param("wa"),
        }

        kbt, tmp_bt = ee2.get_boost2(p, 
                                     self.z, 
                                     ee2.PyEuclidEmulator(), 
                                     10**np.linspace(-2.0589,0.973,self.len_k))  
        logbt = np.log(np.array(tmp_bt, dtype='float64'))
        boost = np.exp(interp1d(np.log(kbt), 
                        logbt, 
                        axis=1,
                        kind='linear', 
                        bounds_error=False,
                        fill_value=(logbt[:,0], logbt[:,-1]), # constant extrapolation
                        assume_sorted=True)(self.logk))
        return boost

    """----------------------------------------------------------------------"""
    # EFT calculations

    def calculate(self, state, want_derived=True, **params):
        # get params that go to additional computations
        # NOTE: CAMB requires omegabh2 and omegach2, provides Omegam
        # Therefore this theory cannot require Omegam

        #------------------------
        # read params from cobaya
        As       = self.provider.get_param("As")
        ns       = self.provider.get_param("ns")
        H0       = self.provider.get_param("H0")
        omegabh2 = self.provider.get_param('omegabh2')
        omegach2 = self.provider.get_param('omegach2')
        mnu      = self.provider.get_param('mnu')
        w0       = self.provider.get_param("w")
        wa       = self.provider.get_param("wa")

        # derived quantities
        h      = H0/100
        As_1e9 = As*1.0e9
        omegab = self.calc_omegab(omegabh2, h)
        omegam = self.calc_omegam(omegabh2, omegach2, h, mnu)

        #------------------------
        # get the growth factors:
        Dz = get_approximate_D(1e-4, As, omegam, omegab, h, ns, mnu, w0, wa, self.a)
        Rz = growth_correction_R(As, omegam, omegab, h, ns, mnu, w0, wa, self.a)
        D2 = ((Dz / Dz[0]) ** 2) * (Rz / Rz[0])
        D1 = np.sqrt(D2)
        D4 = D2*D2
        
        #-------------------------------
        # Compute the EFT loop integrals
        emul_params = np.array([As_1e9, ns, H0, omegab, omegam, w0, wa])
        ytree, y1loop, y2loop, ystoch = self.unified_emulator.predict(emul_params)

        ytree  = ytree[0].numpy().reshape(3,self.len_k)
        y1loop = y1loop[0].numpy().reshape(3,self.len_k)

        p0_0l = ytree[0]
        p0_1l = ytree[1]
        p0_2l = ytree[2]

        p1_1l = y1loop[0]
        p1_2l = y1loop[1]
        pq_2l = y1loop[2]

        p2_2l = y2loop[0].numpy()*(1+p1_1l)
        pstoch = ystoch[0].numpy()

        #-----------------------------
        # compute the EFT counterterms
        c = np.zeros((5, self.len_z))
        if self.reference_nonlin != 'none':
            nonlin_ref = self.external_boost_fcn()

            X_term1_base = -2 * pi2        * self.k2 * p0_2l
            X_term2_base = -2 * pi2        * self.k2 * (p0_1l - p0_2l)
            X_term2_sub  = -2 * pi2        * self.k2 * p1_2l
            X_term3_base = -2 * pi2        * self.k2 * pq_2l
            X_term4_base = -(pi2**2)       * self.k4 * p0_2l
            X_term5_base = 1000 * (pi2**2) * pstoch

            for i in range(self.len_z):
                Di2 = D2[i]
                Di4 = D4[i]
                #Di6 = D6[i]
                bref = nonlin_ref[i]
                
                mask = self.k <= self.k_fit/D1[i]

                Y_all = (bref - (p0_0l + p1_1l*Di2 + p2_2l*Di4)) / bref
                X_all = np.column_stack([
                    (X_term1_base  / bref),
                    ((X_term2_base / bref)) + ((X_term2_sub / bref) * Di2),
                    (X_term3_base  / bref) * Di2,
                    (X_term4_base  / bref),
                    X_term5_base   / (Di2*bref)
                ])
                
                Y = Y_all[mask]
                X = X_all[mask]
                
                # Claude helping optimize:
                #   lstsq is much faster and 
                #   numerically stable than inv(X.T @ X) @ X.T @ Y
                c[:,i] = np.linalg.lstsq(X, Y, rcond=None)[0]

        else:
            # just use a generic fit for the reference counterterm values
            nonlin_ref = np.ones((self.len_z, self.len_k))
            c = np.zeros((5,self.len_z))

        # We add the spline part up to z-max=4. 
        # Beyond that the counterterms are set to the "best fit" values
        # take this as a warning when using probes of matter at z>4!
        cz = np.array([self.provider.get_param('eft_'+str(i)) for i in range(1,46)])
        eft_params = (cz @ self.pca_evecs.T).reshape(5, 9)
        c[:, self.zmask] += eft_params @ self.c_basis

        # Compute the total EFT power spectrum
        term1 = p0_0l[:, None] \
              + p1_1l[:, None] * D2[None, :] \
              + p2_2l[:, None] * D4[None, :]

        term2 = (2 * pi2 * self.k2 * p0_2l)[:, None] * (c[0, :])[None, :]
        term3 = (2 * pi2 * self.k2 * (p0_1l - p0_2l))[:, None] * (c[1, :])[None, :]
        term4 = (2 * pi2 * self.k2 * p1_1l)[:, None] * (c[1, :] * D2)[None, :]
        term5 = (2 * pi2 * self.k2 * pq_2l)[:, None] * (c[2, :] * D2)[None, :]
        term6 = ((pi2**2) * self.k4 * p0_2l)[:, None] * (c[3, :])[None, :]
        term7 = (1000 * (pi2**2) * pstoch)[:, None] * c[4, :][None, :] / D2

        boost = term1 - term2 - term3 - term4 - term5 - term6 + term7

        # blend the boost to the reference boost. This regulates the eft
        highk_sigmoid = scipy.special.expit((self.k[None,:] - (self.k_uv / D1[:, None])) / 0.01)
        
        idx = np.searchsorted(self.k, self.k_uv/D1, side='right') - 1 # finds where k_interp_2D>kuv
        idx = np.clip(idx, 0, self.len_k - 1) # ensures idx is not out of range

        # compute boost value B at each z
        B = boost[idx, np.arange(self.len_z)]/(nonlin_ref.T)[idx, np.arange(self.len_z)]
        uvboost = B[None, :] * np.ones((self.len_k, 1))

        # now we'll set it all up
        boost = ((1-highk_sigmoid) * boost.T + highk_sigmoid * (nonlin_ref * uvboost.T))        

        state["non_linear_ratio"] = {
            'k_h':   self.k,
            'z':     self.z,
            'ratio': np.sqrt(np.abs(boost))
        }

    def get_non_linear_ratio(self, results):
        return self.current_state["non_linear_ratio"]

"""-----------------------------------------------------------------------------
--------------------------------------------------------------------------------
-----------------------------------------------------------------------------"""

"""-----------------------------------------------------------------------------
Original functions from symbolic pofk
  git:
  arXiv:
Numerical safety modifications from V. Lloyd
-----------------------------------------------------------------------------"""

epsilon = 1e-10  # Small value to prevent log(0)

def get_approximate_D(k, As, Om, Ob, h, ns, mnu, w0, wa, a):
    # avoid singularities
    mnu = mnu + 1e-10

    # Get fitting formula without free-streaming
    z = 1 / a - 1
    theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin
    zeq = 2.5e4 * Om * h ** 2 / theta2p7 ** 4

    Omega = Om * a ** (-3)
    OL = (1 - Om) * a ** (-3 * (1 + w0 + wa)) * np.exp(- 3 * wa * (1 - a))
    g = np.sqrt(Omega + OL)
    Omega /= g ** 2
    OL /= g ** 2

    D1 = (
        (1 + zeq) / (1 + z) * 5 * Omega / 2 /
        (Omega ** (4/7) - OL + (1 + Omega/2) * (1 + OL/70))
    )

    # Split Omega_m into CDM, Baryons and Neutrinos
    Onu = mnu / 93.14 / h ** 2
    Oc = Om - Ob - Onu
    fc = Oc / Om
    fb = Ob / Om
    fnu = Onu / Om
    fcb = fc + fb

    # Add Bond et al. 1980 suppression
    pcb = 1/4 * (5 - np.sqrt(1 + 24 * fcb))
    Nnu = (3 if mnu != 0.0 else 0)
    q = k * h * theta2p7 ** 2 / (Om * h ** 2)
    yfs = 17.2 * fnu * (1 + 0.488 / fnu ** (7/6)) * (Nnu * q / fnu) ** 2
    Dcbnu = (fcb ** (0.7/pcb) + (D1 / (1 + yfs)) **
             0.7) ** (pcb / 0.7) * D1 ** (1 - pcb)

    # Remove 1+zeq normalisation given in Eisenstein & Hu 1997
    D = Dcbnu / (1 + zeq)

    return D

def growth_correction_R(As, Om, Ob, h, ns, mnu, w0, wa, a):
    d = np.array([0.8545, 0.394, 0.7294, 0.5347, 0.4662, 4.6669,
                  0.4136, 1.4769, 0.5959, 0.4553, 0.0799, 5.8311,
                  5.8014, 6.7085, 0.3445, 1.2498, 0.3756, 0.2136])

    part1 = d[0]

    log_term1 = np.log(np.abs(-d[5] * w0 - d[6] * wa) + epsilon)
    denominator_inner1 = a * d[1] + d[2] + (Om * d[3] - a * d[4]) * log_term1
    part2 = -1 / denominator_inner1

    log_term2 = np.log(np.abs(-d[9] * w0 - d[10] * wa) + epsilon)
    numerator_inner2 = Om * d[7] - a * d[8] + log_term2

    denominator_inner2 = -a * d[11] + d[12] + d[13] * \
        (Om * d[14] + a * d[15] - 1) * (d[16] * w0 + d[17] * wa + 1)
    part3 = -numerator_inner2 / denominator_inner2

    result = 1 + (1 - a) * (part1 + part2 + part3)

    return result