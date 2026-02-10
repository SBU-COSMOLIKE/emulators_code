import numpy as np
import emcee
import cobaya
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import sys
import os
import scipy
import yaml
from mpi4py import MPI
import copy
from tqdm import tqdm
import time
import argparse

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# Command line args
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='dataset_generator')

parser.add_argument("--yaml", "-y",
										dest="cobaya_yaml",
										help="The training YAML containing the training_args block",
										type=str,
										nargs='?')

parser.add_argument("--probe", "-p",
										dest="probe",
										help="the probe, listed in the yaml, of which to generate data vectors for.",
										type=str,
										nargs='?')

parser.add_argument("--mode", "-m",
										dest="mode",
										help="The generation mode. One of [train, valid, test]",
										type=str,
										nargs='?')

args, unknown = parser.parse_known_args()
cobaya_yaml   = args.cobaya_yaml
probe         = args.probe
mode          = args.mode

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# Class Definition
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
class dataset:
	def __init__(self, cobaya_yaml, probe, mode='train'):
		info = yaml_load(cobaya_yaml)
		self.model = get_model(info)

		with open(cobaya_yaml,'r') as stream:
			args = yaml.safe_load(stream)

		self.sampled_params = args['train_args'][probe]['extra_args']['ord'][0]
		self.prior_params   = list(self.model.parameterization.sampled_params())
		self.sampling_dim   = len(self.sampled_params)
		self.PATH = os.environ.get("ROOTDIR") + '/' + args['train_args']['training_data_path']

		if mode=='train':
			self.N = args['train_args']['n_train']
			self.T = args['train_args']['t_train']
			self.datavectors_file = self.PATH + args['train_args']['train_datavectors_file']
			self.parameters_file  = self.PATH + args['train_args']['train_parameters_file']
		elif mode=='valid':
			self.N = args['train_args']['n_valid']
			self.T = args['train_args']['t_valid']
			self.datavectors_file = self.PATH + args['train_args']['valid_datavectors_file']
			self.parameters_file  = self.PATH + args['train_args']['valid_parameters_file']
		elif mode=='test':
			self.N = args['train_args']['n_test']
			self.T = args['train_args']['t_test']
			self.datavectors_file = self.PATH + args['train_args']['test_datavectors_file']
			self.parameters_file  = self.PATH + args['train_args']['test_parameters_file']

		# Note that the user COULD provide the fiducial cosmology in a different order than cobaya
		self.fiducial = np.zeros(self.sampling_dim)
		for i,param in enumerate(self.sampled_params):
			self.fiducial[i] = args['train_args']['fiducial'][param]

		# get the covariance matrix in the correct order
		raw_covmat = np.loadtxt(args['train_args']['parameter_covmat_file'])
		f = open(args['train_args']['parameter_covmat_file'])
		covmat_params = np.array(f.readline().split(' ')[1:])
		covmat_params[-1] = covmat_params[-1][:-1] # because the last param has a \n
		self.covmat = np.zeros((self.sampling_dim,self.sampling_dim))
		
		for i, param_i in enumerate(self.sampled_params):
			idx1 = np.where(covmat_params == param_i)[0]
			for j,param_j in enumerate(self.sampled_params):
				idx2 = np.where(covmat_params == param_j)[0]
				self.covmat[i,j] = raw_covmat[idx1,idx2]

		# Changes correlation
		sig   = np.sqrt(np.diag(self.covmat))
		outer = np.outer(sig, sig)
		corr  = self.covmat / outer

		m = (lambda A: (np.fill_diagonal(A, 0.0), A.max())[1])(np.abs(corr))
		target = 0.15
		div = (m / target) if (m > target and m > 0.0) else 1.0
		corr /= div
		di = np.diag_indices(corr.shape[0])
		corr[di] = 1.0
		self.covmat = corr * outer
		self.inv_covmat = np.linalg.inv(self.covmat)

	def param_logpost(self,x):
		loglkl   = (-0.5/self.T) * (x-self.fiducial) @ self.inv_covmat @ np.transpose(x-self.fiducial)
		logprior = self.model.prior.logp(x)/self.T
		return loglkl + logprior

	def run_mcmc(self,n_threads=1):
		n_walkers = 100 # we need 100 walkers so when we take 50% burnin and thin by 100, we have N samples

     # now we will get from the model the covmat of the prior to get the initial points
		theta_std = np.diag(self.covmat/25)

		pos0 = self.fiducial[np.newaxis] + 3. * theta_std[np.newaxis] * np.random.normal(size=(n_walkers, self.sampling_dim))

		# with MPIPool() as pool: PARALLEL NOT WORKING NOW
		print('(MCMC) Running parameter space MCMC')
		sampler = emcee.EnsembleSampler(n_walkers, self.sampling_dim, self.param_logpost)#, pool=pool)
		sampler.run_mcmc(pos0, int(5000+2*self.N), progress=True)

		self.samples = sampler.chain.reshape((-1,self.sampling_dim))[(n_walkers*(5000+self.N))::n_walkers]
		N = len(self.samples)

		######The following block is made solely for Roman Bias 1 to 8 ########
		for i in range(1,9):
			self.samples[:,-1*i] = np.random.uniform(0.4,4,N)
		print('(MCMC) Saving parameters to:', self.parameters_file)
		np.savetxt(self.parameters_file, self.samples, header=" ".join(self.sampled_params))

		return True

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	print('INFO')
	print('YAML:',cobaya_yaml)
	print('mode:',mode)
	print('probe:',probe)
	print('')

	if (mode =='all'):
		# train
		generator = dataset(cobaya_yaml, probe, 'train')
		if ( rank == 0 ):
			generator.run_mcmc()						 # generate samples
			print('samples size:',generator.samples.shape)
			#generator.generate_datavectors() # then send main to barrier
		else:
			place_holder = 0
			#generator.generate_datavectors() # send workers to barrier

		# valid
		generator = dataset(cobaya_yaml, probe, 'valid')
		if ( rank == 0 ):
			generator.run_mcmc()
			#generator.generate_datavectors() # then send main to barrier
		else:
			place_holder = 0
			#generator.generate_datavectors() # send workers to barrie

		# test
		generator = dataset(cobaya_yaml, probe, 'test')
		if ( rank == 0 ):
			generator.run_mcmc()
			#generator.generate_datavectors() # then send main to barrier
		else:
			place_holder = 0
			#generator.generate_datavectors() # send workers to barrie
	else:
		# mode
		generator = dataset(cobaya_yaml, probe, mode)
		if ( rank == 0 ):
			generator.run_mcmc()
			#generator.samples = np.loadtxt('./projects/des_y3/des_eft_training_params.txt')[:1000000]
			#generator.samples[:,5] = -1*np.ones(generator.samples.shape[0])
			#generator.samples[:,6] = -1*np.ones(generator.samples.shape[0])
			np.savetxt(generator.parameters_file, generator.samples, header=" ".join(generator.sampled_params))
			#generator.generate_datavectors() # then send main to barrier
		else:
			place_holder = 0
			#generator.generate_datavectors() # send workers to barrie

	MPI.Finalize()
	exit(0)
