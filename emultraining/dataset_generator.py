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
from pathlib import Path

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Command line args
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
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
                    nargs='?',
                    default='train')
parser.add_argument("--chainonly",
                    dest="chainonly",
                    help="Only Compute and output chain with train/test/val params",
                    nargs='?',
                    type=bool,
                    default=True)
args, unknown = parser.parse_known_args()
cobaya_yaml   = args.cobaya_yaml
probe         = args.probe
mode          = args.mode
chainonly     = args.chainonly

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Class Definition
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class dataset:
  #-----------------------------------------------------------------------------
  # init
  #-----------------------------------------------------------------------------  
  def __init__(self, cobaya_yaml, probe, mode='train', target=0.15):
    self.mode  = mode
    self.probe = probe
    #---------------------------------------------------------------------------
    info = yaml_load(cobaya_yaml)
    self.model = get_model(info)
    with open(cobaya_yaml,'r') as stream:
      args = yaml.safe_load(stream)

    #---------------------------------------------------------------------------
    self.sampled_params = args['train_args'][self.probe]['extra_args']['ord'][0]
    self.prior_params   = list(self.model.parameterization.sampled_params())
    self.sampling_dim   = len(self.sampled_params)
    #---------------------------------------------------------------------------
    self.PATH = (os.environ.get("ROOTDIR") + '/' + 
                 args['train_args']['training_data_path'])
    if self.mode == 'train':
      self.N = args['train_args']['n_train']
      self.T = args['train_args']['t_train']
      self.datavectors_file = (self.PATH + 
                               args['train_args']['train_datavectors_file'])
      self.parameters_file  = (self.PATH + 
                               args['train_args']['train_parameters_file'])
    elif self.mode == 'valid':
      self.N = args['train_args']['n_valid']
      self.T = args['train_args']['t_valid']
      self.datavectors_file = (self.PATH + 
                               args['train_args']['valid_datavectors_file'])
      self.parameters_file  = (self.PATH + 
                               args['train_args']['valid_parameters_file'])
    elif self.mode == 'test':
      self.N = args['train_args']['n_test']
      self.T = args['train_args']['t_test']
      self.datavectors_file = (self.PATH + 
                               args['train_args']['test_datavectors_file'])
      self.parameters_file  = (self.PATH + 
                               args['train_args']['test_parameters_file'])

    #---------------------------------------------------------------------------
    # Reorder fiducial data vector
    fid = args["train_args"]["fiducial"]
    self.fiducial = np.array([fid[p] for p in self.sampled_params], dtype=float)

    #---------------------------------------------------------------------------
    # Reorder cov to follow # ['extra_args']['ord']
    with open(args["train_args"]["parameter_covmat_file"]) as f:
      covmat_params = np.array(f.readline().split()[1:])
      raw_covmat = np.loadtxt(f) 
    pidx = {p : i for i, p in enumerate(covmat_params)} # Map param name: idx
    try:
      idx = np.array([pidx[p] for p in self.sampled_params], dtype=int)
    except KeyError as e:
      raise ValueError(f"Param {e.args[0]!r} not found in covmat header") from None
    self.covmat = raw_covmat[np.ix_(idx, idx)]
    
    #---------------------------------------------------------------------------
    # Reorder bounds to follow # ['extra_args']['ord']
    raw_bounds  = model.prior.bounds(confidence=0.999999)
    self.bounds = np.asarray(raw_bounds)[idx, :]    

    #---------------------------------------------------------------------------
    # Reduce correlation on the covariance matrix to max = target
    sig   = np.sqrt(np.diag(self.covmat))
    outer = np.outer(sig, sig)
    corr  = self.covmat / outer
    np.abs(corr - np.eye(n)).max()
    corr /= max(1.0, m / target) if m > 0 else 1.0
    np.fill_diagonal(corr, 1.0)
    self.covmat = corr * outer

    #---------------------------------------------------------------------------
    # Inverse
    C = 0.5 * (self.covmat + self.covmat.T)  # enforce symmetry
    jitt = 0.0
    for _ in range(10):
      try:
        L = np.linalg.cholesky(C + jitt*np.eye(C.shape[0]))
        break
      except np.linalg.LinAlgError:
        scale = np.mean(np.diag(C)) # scale jitt to matrix sz start tiny -> grow
        jitt = (1e-12 if jitt == 0 else jitt * 10) * (scale if scale > 0 else 1.0)
    else:
      raise np.linalg.LinAlgError("could not stabilized cov to SPD w/ jitter")
    I = np.eye(C.shape[0])
    self.covmat = C + jitt*np.eye(C.shape[0])
    self.inv_covmat = np.linalg.solve(L.T, np.linalg.solve(L, I))

  #-----------------------------------------------------------------------------
  # likelihood
  #-----------------------------------------------------------------------------
  def param_logpost(self,x):
    y = x - self.fiducial
    logprior = self.model.prior.logp(x)
    return (-0.5*(y @ self.inv_covmat @ y) + logprior)/self.T

  #-----------------------------------------------------------------------------
  # run mcmc
  #-----------------------------------------------------------------------------
  def run_mcmc(self, nwalkers = 100):
    ndim     = self.sampling_dim
    names    = list(self.sampled_params)
    bds      = self.bounds
    nsteps   = int(5000 + 2*self.N)
    burnin   = int(0.3*nstw)
    thin     = float((nsteps-burnin)*nwalkers)/self.N
    nwalkers = max(nwalkers, 3*ndim)
    sampler = emcee.EnsembleSampler(nwalkers = nwalkers, 
                                    ndim = ndim, 
                                    moves=[(emcee.moves.DEMove(), 0.8),
                                           (emcee.moves.DESnookerMove(), 0.2)],
                                    log_prob_fn = self.param_logpost)
    sampler.run_mcmc(initial_state = self.fiducial[np.newaxis] + 
                                     0.5*np.sqrt(np.diag(self.covmat))*
                                     np.random.normal(size=(nwalkers, ndim)), 
                     nsteps = nstw, 
                     progress=True)
    xf   = sampler.get_chain(flat=True, discard=burnin, thin=thin)
    lnp  = sampler.get_log_prob(flat=True, discard=burnin, thin=thin)
    w    = np.ones((len(xf), 1), dtype='float64')
    chi2 = -2*lnp

    self.samples = xf # we just need the samples to compute the data vector

    # save chain begins --------------------------------------------------------
    stem = Path(self.parameters_file).stem
    root = f"{stem}_{self.probe}_{self.mode}"

    hd=f"nwalkers={nwalkers}\n"
    np.savetxt(f"{root}.1.txt",
               np.concatenate([w, lnp[:,None], xf, chi2[:,None]], axis=1),
               fmt="%.9e",
               header=hd + ''.join(["weights", "lnp"] + names),
               comments="# ")
    

    # save a range files -------------------------------------------------------
    hd = ["weights","lnp"] + names + ["chi2*"]
    rows = [(str(n),float(l),float(h)) for n,l,h in zip(names,bds[:,0],bds[:,1])]
    with open(f"{root}.ranges", "w") as f: 
      f.write(f"# {' '.join(hd)}\n")
      f.writelines(f"{n} {l:.5e} {h:.5e}\n" for n, l, h in rows)

    # save paramname files -----------------------------------------------------
    param_info = self.model.info()['params']
    latex  = [param_info[x]['latex'] for x in names]
    names.append("chi2*")
    latex.append("\\chi^2")
    np.savetxt(f"{root}.paramnames", 
               np.column_stack((names,latex)),
               fmt="%s")

    # save a cov matrix --------------------------------------------------------
    samples = loadMCSamples(f"{root}", settings={'ignore_rows': u'0.0'})
    np.savetxt(f"{root}.covmat",
               np.array(samples.cov(), dtype='float64'),
               fmt="%.9e",
               header=' '.join(names),
               comments="# ")
    return True
  
  #-----------------------------------------------------------------------------
  # datavectors
  #-----------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  print(f"INFO\nYAML: {cobaya_yaml}\nmode: {mode}\nprobe: {probe}\n")

  gen = dataset(cobaya_yaml, probe, mode)
  if (rank == 0):
    generator.run_mcmc()
    if not chainonly:
      generator.generate_datavectors()
  else:
    if not chainonly:
      generator.generate_datavectors()
  MPI.Finalize()
  exit(0)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------