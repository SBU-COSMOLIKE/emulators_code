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

The PCA process is a numerical approximation PCA called incremental PCA. This algorithm requires the user to input the number of PCs they expect and a batch size for the algorithm to process data in batches. 

## emul*.py and PCAdatares.py <a name="emul"></a>
We have several scripts with name emul*.py, corresponding to CMB spectra (trained in TRF or CNN), distance or $C_\ell^{\phi\phi} with ResMLP, $\theta^*$ to $H_0$ mapping with a small ResMLP. For those scripts, the user needs to put in the correct input and output files of both the training and validation sets. They also need to provide the directories to the normalization files and possibly the PCA files if it is a ResMLP. For CMB, we train with a close to cosmic-variance covariance matrix, which also needs to be imported in the beginning of the files. And the user needs to specify the path to where you want to save your model parameters.

The user then needs to specify the dimension, channel number, layer number of the model. Furthermore, the batch-size, training epoch numbers also should be specified here.

Then the user just run the scripts to generate the trained models.

One special file here is the GPrdrag.py, which does not run a pytorch model, but a Gaussian Processing model to generate a small GP model for $r_{\rm drag}$. This script just requires the input and output files of the model, and the normalization files. Then the user just needs to run the script to generate the GP model.
