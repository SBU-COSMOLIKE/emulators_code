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

parser.add_argument("--N",
                    dest="N",
                    help="Number of points requested",
                    type=int,
                    nargs='?',
                    const=1,
                    default=100)
parser.add_argument("--camb_ell_min",
                    dest="camb_ell_min",
                    help="minimum of ell output",
                    type=int,
                    nargs='?',
                    const=1,
                    default=2)
parser.add_argument("--camb_ell_max",
                    dest="camb_ell_max",
                    help="maximum of ell output",
                    type=int,
                    nargs='?',
                    const=1,
                    default=5000)

parser.add_argument("--mode",
                    dest="mode",
                    help="mode of points generated",
                    type=str,
                    nargs='?',
                    const=1,
                    default='train')
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
  planck_2018_highl_plik.TTTEEE_lite:
    path: ./external_modules/
    clik_file: plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik

  planck_2018_lensing.clik:
    path: ./external_modules/
    #clik_file: plc_3.0/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_agr2.clik_lensing
    clik_file: plc_3.0/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing


params:
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
    proposal: 3
    latex: \log(10^{10} A_\mathrm{s})
    drop: true
  
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
  mnu:
    value: 0.06

  A_planck:
    value: 1
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
      halofit_version: mead2020
      #dark_energy_model: ppf
      lmax: 7000
      AccuracyBoost: 1.5
      lens_potential_accuracy: 8
      lens_k_eta_reference: 18000.0
      nonlinear: NonLinear_both
      recombination_model: CosmoRec
      Accuracy.AccurateBB: True



output: ./projects/axions/chains/EXAMPLE_EVALUATE0

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



if __name__ == '__main__':
    model = get_model(yaml_load(yaml_string))
    mode = args.mode

    N = args.N

    prior_params = list(model.parameterization.sampled_params())
    sampling_dim = len(prior_params)
    camb_ell_max = args.camb_ell_max
    camb_ell_min = args.camb_ell_min
    PATH = os.environ.get("ROOTDIR") + '/' + args.data_path
    datavectors_file_path = PATH + args.datavectors_file
    parameters_file  = PATH + args.parameters_file

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print('rank',rank,'is at barrier')
        
    camb_ell_range = camb_ell_max-camb_ell_min
    camb_num_spectra = 4
    CMB_DIR = datavectors_file_path + '_cmb.npy'
    EXTRA_DIR = datavectors_file_path + '_extra.npy'
    u_bound = model.prior.bounds()[:,1]
    l_bound = model.prior.bounds()[:,0]
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
    total_cls = np.zeros(
            (num_datavector, camb_ell_range, camb_num_spectra), dtype = "float32"
        ) 
    extra_dv = np.zeros(
            (num_datavector, 2), dtype = "float32"
        ) 
    for i in range(num_datavector):
        input_params = model.parameterization.to_input(param_info[i])
        input_params.pop("As", None)

        try:
            model.loglike(input_params)
            theory = list(model.theory.values())[1]
            cmb = theory.get_Cl()
            rdrag = theory.get_param("rdrag")
            thetastar = theory.get_param("thetastar")
                
        except:
            print('fail')
        else:
            total_cls[i,:,0] = cmb["tt"][camb_ell_min:camb_ell_max]
            total_cls[i,:,1] = cmb["te"][camb_ell_min:camb_ell_max]
            total_cls[i,:,2] = cmb["ee"][camb_ell_min:camb_ell_max]
            total_cls[i,:,3] = cmb["pp"][camb_ell_min:camb_ell_max]

            extra_dv[i,0] = thetastar
            extra_dv[i,1] = rdrag

    if rank == 0:
        result_cls   = np.zeros((total_num_dvs, camb_ell_range, 4), dtype="float32")
        result_extra = np.zeros((total_num_dvs, 2), dtype="float32") 
        result_cls[0:total_num_dvs:num_ranks] = total_cls ## CMB       
        result_extra[0:total_num_dvs:num_ranks]   = extra_dv ##0: 100theta^*, 1: r_drag

        for i in range(1,num_ranks):        
            result_cls[i:total_num_dvs:num_ranks,:,0] = comm.recv(source = i, tag = 10)
            result_extra[i:total_num_dvs:num_ranks]   = comm.recv(source = i, tag = 12)

        np.save(output_file_cmb, result_cls)
        np.save(output_file_extra, result_extra)
            
    else:    
        comm.send(total_cls[:,:,0], dest = 0, tag = 10)
        comm.send(extra_dv, dest = 0, tag = 12)


#mpirun -n 5 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
#     --bind-to core --map-by core --report-bindings --mca mpi_yield_when_idle 1 \
#    python datageneratorcmb.py \
#    --camb_ell_min 2 \
#    --camb_ell_max 5000 \
#    --data_path './trainingdata/' \
#    --datavectors_file 'dvfilename' \
#    --parameters_file 'paramfilename.npy' \
#    --N 100 \
#    --mode 'train' \
