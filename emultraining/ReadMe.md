## datagenerator.py <a name="overview"></a>
This file shows how to generate generator input paratemer sets in
1. Latin Hypercube sampling,
2. Uniform sample.
The user needs to specify the upper and lower limits, and the number of data vectors they want. Then just run

```
python datagenerator.py
```
# CMBunidv.py <a name="overview"></a>
This file needs the user to specify the input file name. This file will then run an MPI job to generate the
CMBlensed power spectra, hubble parameter and supernovae luminosity distance on the specified redshift, $r_{\rm drag}$
and $100\theta^*$, and linear and nonlinear Matter Power spectra.
The user can use the CMBunidv.sh file to submit those mpi jobs on seawulf with
```
sbatch --array=i-j CMBunidv.sh
```
