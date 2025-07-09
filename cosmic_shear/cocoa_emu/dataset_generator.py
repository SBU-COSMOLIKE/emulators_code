import numpy as np
import emcee
import cobaya
from cobaya.yaml import yaml_load
from cobaya.model import get_model
import sys
import os

sys.path.append(os.path.dirname(__file__))
from config import cocoa_config

class generate_samples:
	def __init__(self, N, T, model, covmatfile):
		self.N = N
		self.T = T
		self.sampling_dim = len(list(model.parameterization.sampled_params()))

		if '.txt' in covmatfile:
			covmat = np.loadtxt(covmatfile)
		else:
			covmat = np.load(covmatfile)

		# we want to take this copy of the model and:
		#	 extend the hardpriors by 5%
		#	 temper the gaussian priors by T
		self.model = model
		self.inv_cov = np.inv(covmat)

	def param_logpost(x):
		loglkl   = x @ inv_cov @ np.transpose(x)
		logprior = self.model.prior.logp(x)
		return loglkl + logprior

	def run_mcmc(n_threads=1):
		n_walkers = int(N*100) # we need 100 walkers so when we thin by 100, we have N samples

		pos0 = theta_fid[np.newaxis] + 3. * theta_std[np.newaxis] * np.random.normal(size=(n_walkers, salf.sampling_dim))

		with mp.Pool(n_threads) as pool:
			sampler = emcee.EnsembleSampler(n_walkers, self.sampling_dim, param_logpost, pool=pool)
			sampler.run_mcmc(pos0, self.N, progress=True)

		samples = sampler.chain.reshape((-1,self.sampling_dim))



def generate_datavectors():
	return True

def generate_dataset(N,T,covmat_file,cobaya_yaml):
	path = './external_modules/data/lsst_y1_cosmic_shear_emulator/training_data/'

	# lets see if we can load in the cobaya model:
	info = yaml_load('./projects/lsst_y1/'+cobaya_yaml)
	model = get_model(info)
	sampled_params = list(model.parameterization.sampled_params())

	print('The parameter order is:')
	print(sampled_params)

	samples = generate_samples(100,128,model,None)

	return True


'''
# !!!  Make sure to change your directories !!! #
shifted_param=sys.argv[1]
sigma=int(sys.argv[2])
T=int(sys.argv[3])
write_file = sys.argv[4]

print('Running chain with T={} and {} shifted {} sigma'.format(T,shifted_param,sigma))
write_dir_root = '/gpfs/projects/MirandaGroup/evan/cocoa/Cocoa/projects/lsst_y1/emulator_output/chains/'
write_dir = write_dir_root+write_file


### Prior and Liklihood
cosmo_prior_lim = np.array([[0.275, 5.225],
                            [0.001, 0.927]]) # the lower bound was not extended by 5% because it is so small already, just shifted the decimal
                                             # upper bound chosen so that Sum(Omega_i)<1

ia_prior_lim = np.array([[-5.5, 5.5],
                       [-5.5, 5.5]])

dz_source_std   = 0.002 * np.ones(5) * T * 100
dz_lens_std     = 0.005 * np.ones(5) * T * 100
shear_calib_std = 0.005 * np.ones(5) * T * 100

def hard_prior(theta, params_prior):
    is_lower_than_min = bool(np.sum(theta < params_prior[:,0]))
    is_higher_than_max = bool(np.sum(theta > params_prior[:,1]))
    if is_lower_than_min or is_higher_than_max:
        return -np.inf
    else:
        return 0.
    
def lnprior(theta):
    cosmo_theta = theta
    
    cosmo_prior = hard_prior(cosmo_theta, cosmo_prior_lim)
    
    return cosmo_prior
    
def ln_lkl(theta):
    diff = theta-means
    lkl = (-0.5/T) * (diff @ inv_cov @ np.transpose(diff))
    return lkl

def lnprob(theta):
    prob=lnprior(theta) + ln_lkl(theta)
    return prob

N_MCMC        = 50000
N_WALKERS     = 120
NDIM_SAMPLING = 2

fiducial  = np.array([[2.1, 0.97, 69.0, 0.048, 0.3]])

theta0    = np.array([2.1,0.3])
mean = theta0

theta_std = np.array([0.05, 0.05])

means  = np.array(mean)

inv_cov = np.loadtxt('projects/lsst_y1/fisher.txt')
inv_cov = np.array([[inv_cov[0,0],inv_cov[0,4]],[inv_cov[4,0],inv_cov[4,4]]])
print(inv_cov)

# Starting position of the emcee chain
pos0 = theta0[np.newaxis] + 3. * theta_std[np.newaxis] * np.random.normal(size=(N_WALKERS, NDIM_SAMPLING))

# parallel sampling
n_cpus = mp.cpu_count()
print('n_cpus = {}'.format(n_cpus))

#write
names = ['As', 'ns', 'H0', 'Omegab', 'Omegam',
         'dz_source1','dz_source2','dz_source3','dz_source4','dz_source5',
         'IA1','IA2']
labels = names

# Do the sampling
with mp.Pool(2*n_cpus) as pool:
    sampler = emcee.EnsembleSampler(N_WALKERS, NDIM_SAMPLING, lnprob, pool=pool)
    sampler.run_mcmc(pos0, N_MCMC, progress=True)

samples = sampler.chain.reshape((-1,NDIM_SAMPLING))
chain = []
for s in samples:
    #print(s)
    chain.append([s[0],0.97,69.0,0.048,s[1],0,0,0,0,0,0.5,0])

chain = np.array(chain)
mcsamples = getdist.mcsamples.MCSamples(samples=chain, names=names, labels=labels)#, ranges=cosmo_prior_lim)
mcsamples.removeBurn(0.5)
mcsamples.thin(60)
mcsamples.saveAsText(write_dir+'_0')

print('Chains written to: '+write_dir)

'''
