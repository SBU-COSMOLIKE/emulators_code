## Overview of the Emulators <a name="overview"></a>

Emulators are Machine Learning models that map cosmological (and nuisance) parameters to cosmological data vectors, just like many numerical packages for corresponding observables. The advantage of emulators is the fast evaluation time. As cosmology moves into an era when intense analyses require more and more computational resources for experiments with advancing precision and resolution, many traditional numerical packages become too slow in terms of evaluation time. Emulators are developed to overcome this issue and can widely replace many numerical packages in MCMCs.

In this repository, we provide emulators for various data vectors and in different architectures. The architectures include Residual Multi-Layer Perceptron, Transformer, Convolutional Neural Network, and Gaussian processing models.

## CMB Emulators <a name="CMB"></a>

Cosmic Microwave Background (CMB) emulators are trained to emulate `TT` (temperature auto spectrum), `TE` (temperature-polarization cross spectrum), and `EE` (polarization auto spectrum) power spectra output calculated by `CAMB` Boltznman code. We provide a few architecture options; for instance, the users can select `modeltype` to be either `TRF` (standing for Transformer) or `CNN` (standing for Convolutional Neural Network).  Users then need to specify the corresponding emulator, each emulator correspond to a cosmological model as well as a specific CAMB version and accuracy settings. This is done as follows

**Step :one:**: Select the 'ordering' list, which needs to be the exact sequence of parameters input to the emulator.,

**Step :two:**: Select the emulator information files as shown below

    #Switch the prefix xy for (tt, te, ee)
    'xyfilename':   emulator model parameters and 
    'xyextraname': normalization factors, . 

**Step :three:**: Specify `ellmax`, which is the largest ell mode that the emulator is trained to evaluate.

**Step :four:**: If users want to sample the parameter `theta_*` (recommended) instead of `H_0`, specify the following parameters

     'thetaordering': # exact sequence of parameters input to the Gaussian Processing emulator.
     'GPfilename':    # Gaussian Processing for the mapping between theta_* and thetaordering params to H_0.
     'GPextraname':   # Gaussian Processing for the mapping between theta_* and thetaordering params to H_0.
     
## SN and BAO Emulators <a name="SNBAO"></a>

For Supernovae luminosity distances and BAO observable emulators, we adopt ResMLP for luminosity distance and Hubble parameter. On top of that, we have a Gaussian processing emulator for $r_{drag}$. The users again need to specify the emulator parameter, normalization factor file, and the corresponding PCA transformation matrix file directories. The specific redshift z grid file is also required. Furthermore, we have a parameter named 'dllevel' which refers to a possible artificial shift to the dataset for distance emulators, which is due to the fact that we need to lift all data points above 0 when there are negative distances for negative redshift during training (for the sake of better interpolation). Thus, we can take the logarithm of the distance to better emulate this function.
