import numpy as np
import cobaya
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import sys
import os
import platform
import yaml
from mpi4py import MPI
from scipy.stats import qmc
import copy
import functools, iminuit, copy, argparse, random, time 
import emcee, itertools
from schwimmbad import MPIPool


parser = argparse.ArgumentParser(prog='cos_uniform')

def list_of_strings(arg):
    return arg.split(',')
def list_of_floats(arg):
    return arg.split(',')

parser.add_argument("--N",
                    dest="N",
                    help="Number of points requested",
                    type=int,
                    nargs='?',
                    const=1,
                    default=100)

parser.add_argument("--mode",
                    dest="mode",
                    help="mode of points generated",
                    type=str,
                    nargs='?',
                    const=1,
                    default='train')
parser.add_argument("--l_bound",
                    dest="l_bound",
                    help="lower bound for parameters",
                    type=list_of_floats,
                    nargs='?',
                    const=1,
                    default=None)
parser.add_argument("--u_bound",
                    dest="u_bound",
                    help="upper bound for parameters",
                    type=list_of_floats,
                    nargs='?',
                    const=1,
                    default=None)
parser.add_argument("--ordering",
                    dest="ordering",
                    help="Ordering of parameters",
                    type=list_of_strings,
                    nargs='?',
                    const=1,
                    default=None)
parser.add_argument("--data_path",
                    dest="data_path",
                    help="Directory for saving the data files",
                    type=str,
                    nargs='?',
                    const=1,
                    default='/data/')
parser.add_argument("--datavectors_file",
                    dest="datavectors_file",
                    help="Name for data vector files (no .npy included)",
                    type=str,
                    nargs='?',
                    const=1,
                    default='cos_uni')
parser.add_argument("--parameters_file",
                    dest="parameters_file",
                    help="Name for Parameter files (.npy included)",
                    type=str,
                    nargs='?',
                    const=1,
                    default='cos_uni_input.npy')
args, unknown = parser.parse_known_args()

####add yaml here, and make all input paratemers passing in
yaml_string=r"""

likelihood:
  dummy.desi_re.desi_re:
    path: ./external_modules/data/

params:
  
  omegabh2:
    prior:
      min: 0.0
      max: 0.04
    ref:
      dist: norm
      loc: 0.022383
      scale: 0.005
    proposal: 0.005
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    prior:
      min: 0.0
      max: 0.5
    ref:
      dist: norm
      loc: 0.12011
      scale: 0.03
    proposal: 0.03
    latex: \Omega_\mathrm{c} h^2
  H0:
    prior:
      min: 20
      max: 120
    ref:
      dist: norm
      loc: 67
      scale: 2
    proposal: 0.001
    latex: H_0
  tau:
    prior:
      min: 0.01
      max: 0.2
    ref:
      dist: norm
      loc: 0.055
      scale: 0.006
    proposal: 0.003
    latex: \tau_\mathrm{reio}

  logA:
    prior:
      min: 1.61
      max: 3.91
    ref:
      dist: norm
      loc: 3.0448
      scale: 0.05
    proposal: 3.04
    latex: \log(10^{10} A_\mathrm{s})
    drop: true
  ns:
    prior:
      min: 0.6
      max: 1.3
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.005
    proposal: 0.005
    latex: n_\mathrm{s}
  As:
    value: 'lambda logA: 1e-10*np.exp(logA)'
    latex: A_\mathrm{s}

  A_planck:
    value: 1
  mnu:
    value: 0.06
  thetastar:
    derived: true
    latex: \Theta_\star
  rdrag:
    derived: True
    latex: r_\mathrm{drag}

theory:
  camb:
    path: ./external_modules/code/CAMB
    extra_args:
      #dark_energy_model: ppf
      lmax: 3000





"""


####add main function.


#===================================================================================================
# datavectors

def generate_parameters(N, u_bound, l_bound, mode, parameters_file, save=True):
    D = len(u_bound)

    if mode=='train':
        
        N_LHS = int(0.05*N)
        sampler = qmc.LatinHypercube(d=D)
        sample = sampler.random(n=N_LHS)
        sample_scaled = qmc.scale(sample, l_bound, u_bound)

        N_uni = N-N_LHS
        data = np.random.uniform(low=l_bound, high=u_bound, size=(N_uni, D))
        samples = np.concatenate((sample_scaled, data), axis=0)
    else:
        samples = np.random.uniform(low=l_bound, high=u_bound, size=(N, D))

    if save:
        np.save(parameters_file, samples)
        print('(Input Parameters) Saved!')
    return samples
mode = args.mode
sampled_params = args.ordering
N = args.N
u_bound = [float(s) for s in args.u_bound]
l_bound = [float(s) for s in args.l_bound]
PATH = os.environ.get("ROOTDIR") + '/' + args.data_path
parameters_file  = PATH + args.parameters_file

if __name__ == '__main__':
    model = get_model(yaml_load(yaml_string))
    
    prior_params = list(model.parameterization.sampled_params())
    sampling_dim = len(sampled_params)
    datavectors_file_path = PATH + args.datavectors_file
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print('rank',rank,'is at barrier')

        
    start = time.time()
        
    z1=np.linspace(0,3,600, endpoint=False)
    z2=np.linspace(3,1200,200)
    z = np.concatenate((z1,z2),axis=0)
    len_z = len(z)
    num_output = 2
    SN_DIR = datavectors_file_path + '_baosn.npy'
    if rank == 0:
        samples = generate_parameters(N, u_bound, l_bound, mode, parameters_file)
        total_num_dvs = len(samples)

        param_info = samples[0:total_num_dvs:num_ranks]#reading for 0th rank input
        for i in range(1,num_ranks):#sending other ranks' data
            comm.send(
                samples[i:total_num_dvs:num_ranks], 
                dest = i, 
                tag  = 1
            )
                
    else:
            
        param_info = comm.recv(source = 0, tag = 1)

            
    num_datavector = len(param_info)


    BAOSN = np.zeros(
            (num_datavector, len_z, num_output), dtype = "float32"
        ) 


    for i in range(num_datavector):
        #model.parameterization.sampled_params()["As"] = 0
        input_params = model.parameterization.to_input(param_info[i])
        
        input_params.pop("As", None)
        #print(model.logposterior(input_params))

        try:
            model.logposterior(input_params)
            theory = list(model.theory.values())[1]
            H = theory.get_Hubble(z)
            Dl = theory.get_angular_diameter_distance(z)*(1+z)**2
                
        except:
            print('fail')
        else:

            BAOSN[i,:,0] = H
            BAOSN[i,:,1] = Dl
            

    if rank == 0:
        result_baosn   = np.zeros((total_num_dvs, len_z, num_output), dtype="float32")

            
        result_baosn[0:total_num_dvs:num_ranks] = BAOSN
            



        for i in range(1,num_ranks):        
            result_baosn[i:total_num_dvs:num_ranks] = comm.recv(source = i, tag = 10)


        np.save(SN_DIR, result_baosn)
            
    else:    
        comm.send(BAOSN, dest = 0, tag = 10)




#mpirun -n 5 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
#     --bind-to core --map-by core --report-bindings --mca mpi_yield_when_idle 1 \
#    python datageneratorbaosn.py \
#    --ordering 'omegabh2','omegach2','H0','tau','logA','ns' \
#    --data_path './trainingdata/' \
#    --datavectors_file 'dvfilename' \
#    --parameters_file 'paramfilename.npy' \
#    --N 100 \
#    --mode 'train' \
#    --u_bound 0.038,0.235,114,0.15,3.6,1.3 \
#    --l_bound 0,0.03,25,0.007,1.61,0.7
