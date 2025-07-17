## datagenerator.py <a name="generate"></a>
This file shows how to generate generator input paratemer sets in
1. Latin Hypercube sampling,
2. Uniform sample.
The user needs to specify the upper and lower limits, and the number of data vectors they want. Then just run

```
python datagenerator.py
```

## CMBunidv.py <a name="mpirun"></a>
This file needs the user to specify the input file name. This file will then run an MPI job to generate the
CMBlensed power spectra, hubble parameter and supernovae luminosity distance on the specified redshift, $r_{\rm drag}$
and $100\theta^*$, and linear and nonlinear Matter Power spectra.
The user can use the CMBunidv.sh file to submit those mpi jobs on Seawulf with
```
sbatch --array=i-j CMBunidv.sh
```

## normalizedata.py and PCAdatares.py <a name="normalization"></a>
The next step is pre-processing of data vectors.

The first step is to do any rescaling of the data vectors that we would like to do. For example, for distance or $C_\ell^{\phi\phi}, 
we take the log of the data vector. For CMB TT TE and EE, we do a rescaling of in terms of $A_se^{-2\tau}$.

The second step is to take the mean and std of the data set in both inputs and outputs pre-processed. For certain data, we also do PCA, then we also need to compute the mean and std of the PCA transformed data vectors. We then save those info in a dictionary, also the PCA transformation matrix if existing.
