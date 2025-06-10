# Table of contents
1. [Overview of the Emulators](#overview)
2. [CMB Emulators](#CMB)
3. [Supernovae and BAO Emulators](#SNBAO)

## Overview of the Emulators <a name="overview"></a>

Emulators are Machine Learning models that map cosmological (and nuisance) parameters to cosmological data vectors, just like many numerical packages for corresponding observables. The advantage of emulators is the fast evaluation time. As cosmology moves into an era when intense analyses require more and more computational resources for experiments with advancing precision and resolution, many traditional numerical packages become too slow in terms of evaluation time. Emulators are developed to overcome this issue and can widely replace many numerical packages in MCMCs.

In this repository, we provide emulators for various data vectors and in different architectures. The architectures include Residual Multi-Layer Perceptron, Transformer, Convolutional Neural Network, and Gaussian processing models.

## CMB Emulators <a name="CMB"></a>

The CMB emulators are trained to emulate TT, TE, and EE power spectra output from CAMB. For architecture, the users can select 'modeltype' to be either `TRF` (standing for Transformer) or `CNN` (Convolutional Neural Network). The users then need to specify the corresponding emulator (for specific CAMB version/CAMB accuracy setting/extension of canonical LCDM CAMB) by selecting the 'ordering' list (which needs to be the exact sequence of parameters input to the emulator), and the emulator information files ('xyfilename' for emulator model parameters and 'xyextraname' for normalization factors, xy=tt te ee). If theta_* is sampled instead of H_0, the user needs to specify the 'thetaordering', 'GPfilename', and 'GPextraname' for a Gaussian Processing model for a mapping between theta_* (and other parameters) to H_0. The user also needs to specify 'ellmax', which is the largest ell mode that the emulator is trained to evaluate.

## SN and BAO Emulators <a name="SNBAO"></a>

For Supernovae luminosity distances and BAO observable emulators, we adopt ResMLP for luminosity distance and Hubble parameter. On top of that, we have a Gaussian processing emulator for $r_{drag}$. The users again need to specify the emulator parameter, normalization factor file, and the corresponding PCA transformation matrix file directories. The specific redshift z grid file is also required. Furthermore, we have a parameter named 'dllevel' which refers to a possible artificial shift to the dataset for distance emulators, which is due to the fact that we need to lift all data points above 0 when there are negative distances for negative redshift during training (for the sake of better interpolation). Thus, we can take the logarithm of the distance to better emulate this function.
