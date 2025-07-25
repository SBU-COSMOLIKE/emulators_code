import numpy as np
import cobaya
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import sys
import os
import yaml
from mpi4py import MPI
from scipy.stats import qmc
import copy
from tqdm import tqdm
import time


comm = MPI.COMM_WORLD

class dataset:
    def __init__(self, cobaya_yaml, mode='train'):
        self.cobaya_yaml = cobaya_yaml
        self.mode = mode
        info = yaml_load(cobaya_yaml)

        self.model = get_model(info)

        # we need so more info from the cobaya yaml: the fiducial cosmology and covmat_file
        with open(self.cobaya_yaml,'r') as stream:
            args = yaml.safe_load(stream)

        self.sampled_params = args['train_args']['extra_args']['ord'][0]
        self.prior_params = list(self.model.parameterization.sampled_params())
        self.camb_ell_min = args['train_args']['extra_args']['camb_ell_min']
        self.camb_ell_max = args['train_args']['extra_args']['camb_ell_max']
        # we need to match the ordering. 
        # We can do this via indexes and just shuffle around the samples 
        # when computing the prior.
        # we also want to check all prior params are in sampled params.
        self.sampling_dim = len(self.sampled_params)

        self.PATH = os.environ.get("ROOTDIR") + '/' + args['train_args']['training_data_path']

        if mode=='train':
            self.N = args['train_args']['n_train']
            self.datavectors_file = self.PATH + args['train_args']['train_datavectors_file']
            self.parameters_file  = self.PATH + args['train_args']['train_parameters_file']
            self.u_bound = args['train_args']['train_u_bound']
            self.l_bound = args['train_args']['train_l_bound']

        elif mode=='valid':
            self.N = args['train_args']['n_valid']
            self.datavectors_file = self.PATH + args['train_args']['valid_datavectors_file']
            self.parameters_file  = self.PATH + args['train_args']['valid_parameters_file']
            self.u_bound = args['train_args']['vali_u_bound']
            self.l_bound = args['train_args']['vali_l_bound']

        elif mode=='test':
            self.N = args['train_args']['n_test']
            self.datavectors_file = self.PATH + args['train_args']['test_datavectors_file']
            self.parameters_file  = self.PATH + args['train_args']['test_parameters_file']
            self.u_bound = args['train_args']['vali_u_bound']
            self.l_bound = args['train_args']['vali_l_bound']



#===================================================================================================
# datavectors

    def generate_parameters(self, save=True):
        if self.mode=='train':
            D = len(self.u_bound)
            N_LHS = int(0.05*self.N)
            sampler = qmc.LatinHypercube(d=D)
            sample = sampler.random(n=N_LHS)
            sample_scaled = qmc.scale(sample, self.l_bound, self.u_bound)

            N_uni = self.N-N_LHS
            data = np.random.uniform(low=self.l_bound, high=self.u_bound, size=(N_uni, D))
            self.samples = np.concatenate((sample_scaled, data), axis=0)
        else:
            self.samples = np.random.uniform(low=self.l_bound, high=self.u_bound, size=(self.N, D))

        if save:
            np.save(self.parameters_file, self.samples)
            print('(Input Parameters) Saved!')




    def generate_datavectors(self, save=True):
        rank = comm.Get_rank()
        num_ranks = comm.Get_size()

        print('rank',rank,'is at barrier')
        z1=np.linspace(0,3,600, endpoint=False)
        z2=np.linspace(3,1200,200)
        z = np.concatenate((z1,z2),axis=0) #this redshift is for BAOSN
        len_z          = len(z)

        #MPS setup
        z1_mps = np.linspace(0,2,100,endpoint=False)
        z2_mps = np.linspace(2,10,10,endpoint=False)
        z3_mps = np.linspace(10,50,12)
        z_mps = np.concatenate((z1_mps,z2_mps,z3_mps),axis=0)
        kh_max = 1e2 #maximum k/h
        kh_min = 1e-4 #minimum k/h
        k_n_points = 2000 #Number of k's
        len_mps = len(z_mps)
        
        start = time.time()
        
        camb_ell_range = self.camb_ell_max-self.camb_ell_min
        camb_num_spectra = 4
        self.CMB_DIR = self.datavectors_file + '_cmb.npy'
        #self.BAOSN_DIR = self.datavectors_file + '_baosn.npy'
        self.EXTRA_DIR = self.datavectors_file + '_extra.npy'
        #self.MPS_LIN_DIR = self.datavectors_file + '_lin_mps.npy'
        #self.MPS_NONLIN_DIR = self.datavectors_file + '_nonlin_mps.npy'
        if rank == 0:
            
            total_num_dvs = len(self.samples)

            param_info = self.samples[0:total_num_dvs:num_ranks]#reading for 0th rank input
            for i in range(1,num_ranks):#sending other ranks' data
                comm.send(
                    self.samples[i:total_num_dvs:num_ranks], 
                    dest = i, 
                    tag  = 1
                )
                
        else:
            
            param_info = comm.recv(source = 0, tag = 1)

            
        num_datavector = len(param_info)


        total_cls = np.zeros(
                (num_datavector, camb_ell_range, camb_num_spectra), dtype = "float32"
            ) 

        #SN_dv = np.zeros(
        #        (num_datavector, len_z, 2), dtype = "float32"
        #    ) 

        extra_dv = np.zeros(
                (num_datavector, 2), dtype = "float32"
            ) 

        #PK_lin = np.zeros(
        #        (num_datavector, len_mps, k_n_points), dtype = "float32"
        #    )

        #PK_nonlin = np.zeros(
        #        (num_datavector, len_mps, k_n_points), dtype = "float32"
        #    )

        for i in range(num_datavector):
            print(self.model.parameterization.sampled_params())
            input_params = self.model.parameterization.to_input(param_info[i])
            print(input_params)
            input_params.pop("As", None)
            self.model.logposterior(input_params)
            theory = list(self.model.theory.values())[1]
            print(theory)

            try:
                cmb = theory.get_Cl()
                #dL = theory.get_angular_diameter_distance(z)*(1+z)
                #H = theory.get_Hubble(z, units='km/s/Mpc')
                rdrag = theory.get_param("rdrag")
                thetastar = theory.get_param("thetastar")
                #p_lin=theory.get_Pk_interpolator(nonlinear=False,extrap_kmin=kh_min,extrap_kmax=kh_max)
                #p_nonlin=theory.get_Pk_interpolator(nonlinear=True,extrap_kmin=kh_min,extrap_kmax=kh_max)
                
            except:
                print('fail')
            else:

                total_cls[i,:,0] = cmb["tt"][self.camb_ell_min:self.camb_ell_max]
                total_cls[i,:,1] = cmb["te"][self.camb_ell_min:self.camb_ell_max]
                total_cls[i,:,2] = cmb["ee"][self.camb_ell_min:self.camb_ell_max]
                total_cls[i,:,3] = cmb["pp"][self.camb_ell_min:self.camb_ell_max]


                #SN_dv[i,:,0] = H
                #SN_dv[i,:,1] = dL

                extra_dv[i,0] = thetastar
                extra_dv[i,1] = rdrag

                #PK_lin[i] = p_lin
                #PK_nonlin[i] = p_nonlin

        if rank == 0:
            result_cls   = np.zeros((total_num_dvs, camb_ell_range, 4), dtype="float32")
            #result_sn    = np.zeros((total_num_dvs, len_z, 2), dtype="float32")
            result_extra = np.zeros((total_num_dvs, 2), dtype="float32")
            #result_pklin = np.zeros((total_num_dvs, len_mps, k_n_points), dtype="float32")
            #result_pknonlin = np.zeros((total_num_dvs, len_mps, k_n_points), dtype="float32")
            
            result_cls[0:total_num_dvs:num_ranks] = total_cls ## CMB

            #result_sn[0:total_num_dvs:num_ranks]      = SN_dv            ##0: H, 1: D_l
            
            result_extra[0:total_num_dvs:num_ranks]   = extra_dv         ##0: 100theta^*, 1: r_drag

            #result_pklin[0:total_num_dvs:num_ranks]   = PK_lin           ##P_k linear

            #result_pknonlin[0:total_num_dvs:num_ranks]   = PK_nonlin     ##P_k nonlinear

            for i in range(1,num_ranks):        
                result_cls[i:total_num_dvs:num_ranks,:,0] = comm.recv(source = i, tag = 10)

                #result_sn[i:total_num_dvs:num_ranks]      = comm.recv(source = i, tag = 11)

                result_extra[i:total_num_dvs:num_ranks]   = comm.recv(source = i, tag = 12)

                #result_pklin[i:total_num_dvs:num_ranks]   = comm.recv(source = i, tag = 13)

                #result_pknonlin[i:total_num_dvs:num_ranks]   = comm.recv(source = i, tag = 14)
                
            if save:
                np.save(output_file_cmb, result_cls)
                #np.save(output_file_sn, result_sn)
                np.save(output_file_extra, result_extra)
                #np.save(output_file_pklin, result_pklin)
                #np.save(output_file_pknonlin, result_pknonlin)
            
        else:    
            comm.send(total_cls[:,:,0], dest = 0, tag = 10)

            #comm.send(SN_dv, dest = 0, tag = 11)

            comm.send(extra_dv, dest = 0, tag = 12)

            #comm.send(PK_lin, dest = 0, tag = 13)

            #comm.send(PK_nonlin, dest = 0, tag = 14)



        return True

