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

#===================================================================================================
# Command line args
parser = argparse.ArgumentParser(prog='dataset_generator')

# yaml, probe, and mode:
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

# Sampling arguments #
# parser.add_argument("--n_train",
#                     dest="n_train",
#                     help="The number of training points",
#                     type=int,
#                     nargs='?',
#                     const=1,
#                     default=None)

# need to use parse_known_args because of mpifuture 
args, unknown = parser.parse_known_args()
cobaya_yaml   = args.cobaya_yaml
probe         = args.probe
mode          = args.mode

# Begin program ===================================================================================#
class dataset:
	def __init__(self, cobaya_yaml, probe, mode='train'):
		self.cobaya_yaml = cobaya_yaml

		info = yaml_load(cobaya_yaml)
		self.model = get_model(info)

		# we need so more info from the cobaya yaml: the fiducial cosmology and covmat_file
		with open(self.cobaya_yaml,'r') as stream:
			args = yaml.safe_load(stream)

		self.sampled_params = args['train_args'][probe]['extra_args']['ord'][0]
		self.prior_params = list(self.model.parameterization.sampled_params())
		# we need to match the ordering. 
		# We can do this via indexes and just shuffle around the samples 
		# when computing the prior.
		# we also want to check all prior params are in sampled params.
		self.sampling_dim = len(self.sampled_params)

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

		# construct the fiducial cosmology.
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
		
		for i,param_i in enumerate(self.sampled_params):
			idx1 = np.where(covmat_params==param_i)[0]
			
			for j,param_j in enumerate(self.sampled_params):

				idx2 = np.where(covmat_params==param_j)[0]
				
				self.covmat[i,j] = raw_covmat[idx1,idx2]


		#####The following block changes correlation to some target#####
		sig   = np.sqrt(np.diag(self.covmat))
		outer = np.outer(sig, sig)
		corr  = self.covmat / outer

		
		m = (lambda A: (np.fill_diagonal(A, 0.0), A.max())[1])(np.abs(corr))

		target = 0.15
		div = (m / target) if (m > target and m > 0.0) else 1.0
		corr /= div
		di = np.diag_indices(corr.shape[0])
		corr[di] = 1.0

		self.covmat =corr * outer

		self.inv_covmat = np.linalg.inv(self.covmat)

#===================================================================================================
# broken fisher code

	# def fisher(self):
	# 	raise NotImplementedError
	# 	print("(Fisher) Evaluating Fisher matrix.")
	# 	print("(WARNING) One should explicitly check that the covariance derived from this Fisher matrix \n\
	# covers the paramter space as desired! Hence this script will exit after the covariance \n\
	# is computed. Check the Jupyter notebook (name) on how to check and modify the \n\
	# covariance as needed.")

	# 	# define a step size in percent of prior
	# 	step_size = 0.1 # 10%
	# 	self.samples = np.zeros((1+self.sampling_dim*4,self.sampling_dim))

	# 	priors = self.model.prior.bounds(confidence_for_unbounded=0.9999995) # 5 sigma

	# 	print('(Fisher) computing parameter with step size =',step_size,'of its prior.')

	# 	self.samples[0] = self.fiducial
	# 	for i in range(self.sampling_dim):
	# 		h_1 = (priors[i,1] - priors[i,0])*step_size
	# 		print(h_1)

	# 		p1_1 = copy.deepcopy(self.fiducial)
	# 		p1_2 = copy.deepcopy(self.fiducial)
	# 		p1_3 = copy.deepcopy(self.fiducial)
	# 		p1_4 = copy.deepcopy(self.fiducial)

	# 		p1_1[i] = self.fiducial[i] - 2*h_1
	# 		p1_2[i] = self.fiducial[i] - h_1
	# 		p1_3[i] = self.fiducial[i] + h_1
	# 		p1_4[i] = self.fiducial[i] + 2*h_1

	# 		self.samples[1+4*i] = p1_1
	# 		self.samples[2+4*i] = p1_2
	# 		self.samples[3+4*i] = p1_3
	# 		self.samples[4+4*i] = p1_4

	# 	print('(Fisher) computing datavectors.')
	# 	self.generate_datavectors(save=False)

	# 	print('(Fisher) evaluating matrix.')
	# 	fisher = np.zeros((self.sampling_dim,self.sampling_dim))

	# 	# if we are dealing with cosmic shear, we don't want the galaxy-galaxy lensing and galaxy
	# 	# clustering parts of the covmat, so lets remove them here
	# 	# zero_idxs = np.where(self.datavectors[0]==0)

	# 	# cov_inv = copy.deepcopy(self.config.cov_inv)
	# 	# print(zero_idxs)
	# 	# cov_inv[zero_idxs][:,zero_idxs] = 0

	# 	for i in range(self.sampling_dim):
	# 		for j in range(i,self.sampling_dim):
	# 			h_1 = (priors[i,1] - priors[i,0])*step_size
	# 			h_2 = (priors[j,1] - priors[j,0])*step_size

	# 			dv1_1 = self.datavectors[4*i+1]
	# 			dv1_2 = self.datavectors[4*i+2]
	# 			dv1_3 = self.datavectors[4*i+3]
	# 			dv1_4 = self.datavectors[4*i+4]

	# 			dv2_1 = self.datavectors[4*j+1]
	# 			dv2_2 = self.datavectors[4*j+2]
	# 			dv2_3 = self.datavectors[4*j+3]
	# 			dv2_4 = self.datavectors[4*j+4]

	# 			derivative_1 = (-dv1_4 + 8.0*dv1_3 - 8.0*dv1_2 + dv1_1)/(12.0*h_1)
	# 			derivative_2 = (-dv2_4 + 8.0*dv2_3 - 8.0*dv2_2 + dv2_1)/(12.0*h_2)

	# 			diff_1 = derivative_1[self.config.mask]#-self.datavectors[0][self.config.mask]
	# 			diff_2 = derivative_2[self.config.mask]#-self.datavectors[0][self.config.mask]

	# 			fisher[i,j] = diff_1 @ self.config.cov_inv_masked @ diff_2
	# 			fisher[j,i] = fisher[i,j]

	# 			if( i==j ):
	# 				print(fisher[i,j])
	# 		print(diff_1[:3],diff_1[-3:])
		
	# 	eig = np.linalg.eigh(np.linalg.inv(fisher))
	# 	eig2 = np.linalg.eigh(fisher)
	# 	if np.any(eig[0]<0):
	# 		print('Fisher matrix is not positive semi-definite!')
	# 	if np.any(eig2[0]<0):
	# 		print('not a problem with numpy matrix inverse.')

	# 	print('(Fisher) Saving Fisher matrix to:',self.covmat_file)
	# 	np.savetxt(self.covmat_file, np.linalg.inv(fisher))
	# 	self.inv_cov = fisher

	# 	return True

#===================================================================================================
# MCMC

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
	
#===================================================================================================
# datavectors

	def generate_datavectors(self, save=True):
		rank = comm.Get_rank()
		size = comm.Get_size()

		print('rank',rank,'is at barrier')
		comm.Barrier()
		start = time.time()
		likelihood = self.model.likelihood[list(self.model.likelihood.keys())[0]]

		if( size != 1 ):
			if ( rank == 0 ):
				# i want to get the datavector size. Make this flexible = do one computation beforehand.
				input_params = self.model.parameterization.to_input(self.samples[0])

				self.model.provider.set_current_input_params(input_params)

				for (component, like_index), param_dep in zip(self.model._component_order.items(),
			                                                      self.model._params_of_dependencies):

					depend_list = [input_params[p] for p in param_dep]
					params = {p: input_params[p] for p in component.input_params}
					compute_success = component.check_cache_and_compute(
			            params, want_derived={},
			            dependency_params=depend_list, cached=False)

				datavector = likelihood.get_datavector(**input_params)

				self.datavectors = np.zeros((len(self.samples),len(datavector)),dtype='float32')

				# rank 0 is a manager. It distributes the computations to the workers with rank > 0
				# initialize
				num_sent = 0
				loop_arr = np.arange(0,len(self.samples),1,dtype=int)

				#send the initial data
				print('(Datavectors) Begin computing datavectors...')
				sys.stdout.flush()
				for i in tqdm(range(0,len(self.samples))):
					sys.stdout.flush()
					status = MPI.Status()

					if i in range(0,min(size-1,len(self.samples)-1)):
						comm.send([loop_arr[i],self.samples[i]], dest=i+1, tag=1)

					else:
						idx,datavector = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
						self.datavectors[idx] = datavector
						compute_rank = status.Get_source()
						comm.send([loop_arr[i],self.samples[i]],dest=compute_rank,tag=1)
					sys.stdout.flush()

				# for i in range(0,min(size-1,len(self.samples)-1)):
				# 	comm.send([loop_arr[i],self.samples[i]], dest=i+1, tag=1)
				# 	num_sent += 1

				# # send more data to worker after the worker finishes
				# status = MPI.Status()
				# for i in tqdm(range(num_sent,len(self.samples))):
				# 	print(i,'from main!')
				# 	idx,datavector = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
				# 	self.datavectors[idx] = datavector
				# 	compute_rank = status.Get_source()
				# 	comm.send([loop_arr[i],self.samples[i]],dest=compute_rank,tag=1)

				#communicate to workers that everything is done
				for i in range(1,size):
					idx,datavector = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
					self.datavectors[idx] = datavector
					compute_rank = status.Get_source()
					# print('sending stop to rank',compute_rank)
					comm.send([0,self.samples[0]],dest=compute_rank,tag=0)

				# barrier to wait signal workers to to move forard.
				comm.Barrier()

				if save:
					np.save(self.datavectors_file, self.datavectors)
				print('(Datavectors) Done computing datavectors!')

			else:
				# anything not rank=0 is a worker. It recieves the index of the sample to compute.
				# Each worker will return its index to the manager so that it can recieve 
				# the next available index. The manager always sends with tag=1 unless all computations 
				# have already been distributed, in which case it will send tag=0.

				status = MPI.Status()
				while ( True ):
					# get the information from the manager
					idx,sample = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
					print('idx =',idx,'| rank =',rank,'| runtime =',time.time()-start)
					#print('')

					# check if there is work to be done
					if ( status.Get_tag()==0 ):
						# if work is done, barrier to wait for other workers
						comm.Barrier()
						break

					# else do the work
					print(rank, sample)
					input_params = self.model.parameterization.to_input(sample)
					self.model.provider.set_current_input_params(input_params)

					for (component, like_index), param_dep in zip(self.model._component_order.items(),
				                                                      self.model._params_of_dependencies):

						depend_list = [input_params[p] for p in param_dep]
						params = {p: input_params[p] for p in component.input_params}
						compute_success = component.check_cache_and_compute(
			                params, want_derived={},
			                dependency_params=depend_list, cached=False)

					datavector = likelihood.get_datavector(**input_params)

					comm.send([idx,datavector], dest=0, tag=rank)

		else:
			input_params = self.model.parameterization.to_input(self.samples[0])
			self.model.provider.set_current_input_params(input_params)

			for (component, like_index), param_dep in zip(self.model._component_order.items(),
		                                                      self.model._params_of_dependencies):

				depend_list = [input_params[p] for p in param_dep]
				params = {p: input_params[p] for p in component.input_params}
				compute_success = component.check_cache_and_compute(
		            params, want_derived={},
		            dependency_params=depend_list, cached=False)

			datavector = likelihood.get_datavector(**input_params)
			self.datavectors = np.zeros((len(self.samples),len(datavector)))

			for idx in tqdm(range(len(self.samples))):
				input_params = self.model.parameterization.to_input(self.samples[idx])
				self.model.provider.set_current_input_params(input_params)

				for (component, like_index), param_dep in zip(self.model._component_order.items(),
			                                                      self.model._params_of_dependencies):

					depend_list = [input_params[p] for p in param_dep]
					params = {p: input_params[p] for p in component.input_params}
					compute_success = component.check_cache_and_compute(
		                params, want_derived={},
		                dependency_params=depend_list, cached=False)

				self.datavectors[idx] = likelihood.get_datavector(**input_params)

		return True

#===================================================================================================
# main

if __name__ == "__main__":
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	print('INFO')
	print('YAML:',cobaya_yaml)
	print('mode:',mode)
	print('probe:',probe)
	print('')

	if ( mode =='all' ):
		# train
		generator = dataset(cobaya_yaml, probe, 'train')
		if ( rank == 0 ):
			generator.run_mcmc()						 # generate samples
			print('samples size:',generator.samples.shape)
			generator.generate_datavectors() # then send main to barrier
		else:
			generator.generate_datavectors() # send workers to barrier

		# valid
		generator = dataset(cobaya_yaml, probe, 'valid')
		if ( rank == 0 ):
			generator.run_mcmc()
			generator.generate_datavectors()
		else:
			generator.generate_datavectors()

		# test
		generator = dataset(cobaya_yaml, probe, 'test')
		if ( rank == 0 ):
			generator.run_mcmc()
			generator.generate_datavectors()
		else:
			generator.generate_datavectors()

	else:
		# mode
		generator = dataset(cobaya_yaml, probe, mode)
		if ( rank == 0 ):
			generator.run_mcmc()
			#generator.samples = np.loadtxt('./projects/des_y3/des_eft_training_params.txt')[:1000000]
			#generator.samples[:,5] = -1*np.ones(generator.samples.shape[0])
			#generator.samples[:,6] = -1*np.ones(generator.samples.shape[0])
			np.savetxt(generator.parameters_file, generator.samples, header=" ".join(generator.sampled_params))
			generator.generate_datavectors()
		else:
			generator.generate_datavectors() 

	MPI.Finalize()
	exit(0)

# def generate_dataset(cobaya_yaml, probe, mode, samples_only=False):
# 	# USE samples_only=True IF YOU WANT TO TEST THE FISHER MATRIX

# 	# generate the training samples
# 	generator = dataset(cobaya_yaml,probe,mode)
# 	rank = comm.Get_rank()

# 	if ( rank == 0 ):
# 		#generator.fisher()
# 		generator.run_mcmc()
# 		#print('main going to next loop')
# 		generator.generate_datavectors() # for the MCMC samples

# 	else:
# 		#generator.generate_datavectors() # This one is for fisher
# 		#print('going to next generate loop')
# 		generator.generate_datavectors() # this on is for the MCMC samples

# 	return True
	
