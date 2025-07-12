import numpy as np
from mpi4py import MPI
import scipy
import sys, os
import camb
import scipy.linalg
from camb import model, initialpower
from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid
#import time
if "-f" in sys.argv:
    idx = sys.argv.index('-f')
n= int(sys.argv[idx+1])

comm      = MPI.COMM_WORLD
rank      = comm.Get_rank()
num_ranks = comm.Get_size()

file_name       = 'cosuni_'+str(n)
cosmology_file  = file_name + '.npy'
output_file_cmb     = 'cos_uni_'+str(n) + '_cmb_output' + '.npy'
output_file_sn      = 'cos_uni_'+str(n) + '_sn_output' + '.npy'
output_file_extra   = 'cos_uni_'+str(n) + '_thetardrag_output' + '.npy'
output_file_pklin   = 'cos_uni_'+str(n) + '_pklin_output' + '.npy'
output_file_pknonlin   = 'cos_uni_'+str(n) + '_pknonlin_output' + '.npy'
output_file_sigma8 = 'cos_uni_'+str(n) + '_pknonlin_output' + '.npy'

#CMB setup
camb_accuracy_boost   = 1.5#test 2.5
camb_l_sampple_boost  = 10# test50    # 50 = every ell is computed
camb_ell_min          = 2#30
camb_ell_max          = 10000
camb_ell_range        = camb_ell_max  - camb_ell_min 
camb_num_spectra      = 4

#BAOSN setup
z = np.arange(0,3,0.005) #this redshift is for BAOSN
len_z          = len(z)

#MPS setup
z_mps = np.arange(0,30,1500) #MPS redshift grid
k_max = 1e2 #maximum k
k_min = 1e-4 #minimum k
k_n_points = 2000 #Number of k's
len_mps = len(z_mps)

if rank == 0:
    #start=time.time()
    param_info_total = np.load(
        cosmology_file,
        allow_pickle = True
    )
    total_num_dvs = len(param_info_total)

    param_info = param_info_total[0:total_num_dvs:num_ranks]#reading for 0th rank input
    for i in range(1,num_ranks):#sending other ranks' data
        comm.send(
            param_info_total[i:total_num_dvs:num_ranks], 
            dest = i, 
            tag  = 1
        )
        
else:
    
    param_info = comm.recv(source = 0, tag = 1)

    
num_datavector = len(param_info)


total_cls = np.zeros(
        (num_datavector, camb_ell_range, camb_num_spectra), dtype = "float32"
    ) 
lensing_cl = np.zeros(
        (num_datavector, camb_ell_range), dtype = "float32"
    ) 
SN_dv = np.zeros(
        (num_datavector, len_z, 2), dtype = "float32"
    ) 

extra_dv = np.zeros(
        (num_datavector, 2), dtype = "float32"
    ) 

PK_lin = np.zeros(
        (num_datavector, len_mps, k_n_points), dtype = "float32"
    )

PK_nonlin = np.zeros(
        (num_datavector, len_mps, k_n_points), dtype = "float32"
    ) 





camb_params = camb.CAMBparams()

for i in range(num_datavector):

    camb_params.set_cosmology(
        H0      = param_info[i,2], 
        ombh2   = param_info[i,0],
        omch2   = param_info[i,1], 
        mnu     = 0.06, 
        tau     = param_info[i,3],
        omk     = 0
    )

    camb_params.InitPower.set_params(
        As = np.exp(param_info[i,4])/(1e10), 
        ns = param_info[i,5]
    )
    
    camb_params.set_for_lmax(15000)
    
    camb_params.DarkEnergy = DarkEnergyPPF(
        w = param_info[i,6], 
        wa = param_info[i,7]
    )

    camb_params.set_accuracy(
        AccuracyBoost  = camb_accuracy_boost, 
        lSampleBoost   = camb_l_sampple_boost, 
        lAccuracyBoost = 3,  # we wont change the number of multipoles in the hierarchy
        DoLateRadTruncation = False
    )

    camb_params.set_matter_power(redshifts=z_mps, kmax=2e2)

    try:
        camb_params.NonLinear = model.NonLinear_none
        results_lin = camb.get_results(camb_params)
        pk_linear = results_lin.get_matter_power_spectrum(minkh=k_min, maxkh=k_max, npoints=k_n_points)[2]
        

        camb_params.NonLinear = camb.model.NonLinear_both
        camb_params.NonLinearModel.set_params("mead2016")

        results = camb.get_results(camb_params)

        pk_nonlinear = results.get_matter_power_spectrum(minkh=k_min, maxkh=k_max, npoints=k_n_points)[2]
        powers  = results.get_cmb_power_spectra(
            camb_params, 
            CMB_unit = 'muK',
            raw_cl   = True
        )
        lensing_powers = results.get_lens_potential_cls(
            lmax   = 15000,
            raw_cl = True)
        bg_results = camb.get_background(camb_params)
        extra = bg_results.get_derived_params()
        
    except:
        
        total_cls[i] = np.ones((camb_ell_range, camb_num_spectra)) # put 1s for all   
        lensing_cl[i] = np.ones(camb_ell_range)
        SN_dv[i] = np.ones((len_z, 2))
        extra_dv[i] = np.ones(2)
        PK_lin[i] = np.ones((len_mps, k_n_points))
        PK_nonlin[i] = np.ones((len_mps, k_n_points))

    else:

        total_cls[i] = powers['total'][camb_ell_min:camb_ell_max]
        lensing_cl[i] = lensing_powers[camb_ell_min:camb_ell_max, 0]
        SN_dv[i,:,0] = results.hubble_parameter(z)
        SN_dv[i,:,1] = results.luminosity_distance(z)
        extra_dv[i,0] = extra["thetastar"]
        extra_dv[i,1] = extra["rdrag"]
        PK_lin[i] = pk_linear
        PK_nonlin[i] = pk_nonlinear





if rank == 0:
    result_cls   = np.zeros((total_num_dvs, camb_ell_range, 4), dtype="float32")
    result_sn    = np.zeros((total_num_dvs, len_z, 2), dtype="float32")
    result_extra = np.zeros((total_num_dvs, 2), dtype="float32")
    result_pklin = np.zeros((total_num_dvs, len_mps, k_n_points), dtype="float32")
    result_pknonlin = np.zeros((total_num_dvs, len_mps, k_n_points), dtype="float32")
    
    result_cls[0:total_num_dvs:num_ranks,:,0] = total_cls[:,:,0] ## TT

    result_cls[0:total_num_dvs:num_ranks,:,1] = total_cls[:,:,3] ## TE
        
    result_cls[0:total_num_dvs:num_ranks,:,2] = total_cls[:,:,1] ## EE

    result_cls[0:total_num_dvs:num_ranks,:,3] = lensing_cl       ## phiphi

    result_sn[0:total_num_dvs:num_ranks]      = SN_dv            ##0: H, 1: D_l
    
    result_extra[0:total_num_dvs:num_ranks]   = extra_dv         ##0: 100theta^*, 1: r_drag

    result_pklin[0:total_num_dvs:num_ranks]   = PK_lin           ##P_k linear

    result_pknonlin[0:total_num_dvs:num_ranks]   = PK_nonlin     ##P_k nonlinear

    for i in range(1,num_ranks):        
        result_cls[i:total_num_dvs:num_ranks,:,0] = comm.recv(source = i, tag = 10)
        
        result_cls[i:total_num_dvs:num_ranks,:,1] = comm.recv(source = i, tag = 11)
        
        result_cls[i:total_num_dvs:num_ranks,:,2] = comm.recv(source = i, tag = 12)

        result_cls[i:total_num_dvs:num_ranks,:,3] = comm.recv(source = i, tag = 13)

        result_sn[i:total_num_dvs:num_ranks]      = comm.recv(source = i, tag = 14)

        result_extra[i:total_num_dvs:num_ranks]   = comm.recv(source = i, tag = 15)

        result_pklin[i:total_num_dvs:num_ranks]   = comm.recv(source = i, tag = 16)

        result_pknonlin[i:total_num_dvs:num_ranks]   = comm.recv(source = i, tag = 17)
        

    np.save(output_file_cmb, result_cls)
    np.save(output_file_sn, result_sn)
    np.save(output_file_extra, result_extra)
    np.save(output_file_pklin, result_pklin)
    np.save(output_file_pknonlin, result_pknonlin)
    
else:    
    comm.send(total_cls[:,:,0], dest = 0, tag = 10)
    
    comm.send(total_cls[:,:,3], dest = 0, tag = 11)
    
    comm.send(total_cls[:,:,1], dest = 0, tag = 12)

    comm.send(lensing_cl, dest = 0, tag = 13)

    comm.send(SN_dv, dest = 0, tag = 14)

    comm.send(extra_dv, dest = 0, tag = 15)

    comm.send(PK_lin, dest = 0, tag = 16)

    comm.send(PK_nonlin, dest = 0, tag = 17)

