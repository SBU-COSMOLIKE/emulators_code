import numpy as np
import emcee, argparse, os, sys, scipy, yaml, time
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from mpi4py import MPI
from tqdm import tqdm
from pathlib import Path
from getdist import loadMCSamples
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#mpirun -n 2 --oversubscribe \
#  python external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py \
#    --root projects/roman_real/  \
#    --fileroot emulators/w0wa_nla_halofit_cosmic_shear_cnn/ \
#    --nparams 1000 \
#    --temp 128 \
#    --yaml 'w0wa_takahashi_nobaryon_cs_CNN.yaml' \
#    --datavsfile 'w0wa_takahashi_nobaryon_dvs_train' \
#    --paramfile 'w0wa_params_train' \
#    --chain 1 
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Command line args
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='dataset_generator')

parser.add_argument("--yaml",
                    dest="yaml",
                    help="The training YAML containing the training_args block",
                    type=str,
                    nargs='?')
parser.add_argument("--root",
                    dest="root",
                    help="Project folder",
                    type=str,
                    nargs='?',
                    default="projects/example/")
parser.add_argument("--fileroot",
                    dest="fileroot",
                    help="Subfolder of Project folder where we find yaml and fisher",
                    type=str,
                    nargs='?',
                    default="projects/example/emulators")
parser.add_argument("--mode",
                    dest="mode",
                    help="generation mode = [train, valid, test]",
                    type=str,
                    nargs='?',
                    default='train')
parser.add_argument("--chain",
                    dest="chain",
                    help="only compute and output train/test/val chain",
                    nargs='?',
                    type=bool,
                    default=True)
parser.add_argument("--nparams",
                    dest="nparams",
                    help="Number of Parameters to Generate",
                    nargs='?',
                    type=int,
                    default=100000)
parser.add_argument("--temp",
                    dest="temp",
                    help="Number of Parameters to Generate",
                    nargs='?',
                    type=int,
                    default=128)
parser.add_argument("--datavsfile",
                    dest="datavsfile",
                    help="File to save data vectors",
                    nargs='?',
                    type=str)
parser.add_argument("--paramfile",
                    dest="paramfile",
                    help="File to save parameters",
                    nargs='?',
                    type=str)
args, unknown = parser.parse_known_args()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Class Definition
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class dataset:
  #-----------------------------------------------------------------------------
  # init
  #-----------------------------------------------------------------------------  
  def __init__(self, target=0.15):
    #---------------------------------------------------------------------------
    # Basic definitions
    #---------------------------------------------------------------------------
    root = os.environ.get('ROOTDIR').rstrip('/')
    root = f"{root}/{args.root.rstrip('/')}"
    fileroot = f"{root}/{args.fileroot.rstrip('/')}"
    #---------------------------------------------------------------------------
    # Load Cobaya model (needed for computing likelihood)
    #---------------------------------------------------------------------------
    info = yaml_load(f"{fileroot}/{args.yaml}")
    self.model = get_model(info)
    #---------------------------------------------------------------------------
    # Load yaml again (for reading options)
    #---------------------------------------------------------------------------
    with open(f"{fileroot}/{args.yaml}", 'r') as stream:
      yamlopts = yaml.safe_load(stream)
    
    # preferred ordering of params
    self.sampled_params = yamlopts['train_args']['ord'][0]
    
    # load fiducial data vector
    fid = yamlopts["train_args"]["fiducial"]
    
    # load cov param matrix
    raw_covmat_file = yamlopts["train_args"]["params_covmat_file"]
    with open(f"{fileroot}/{raw_covmat_file}") as f:
      raw_covmat_params_names = np.array(f.readline().split()[1:])
      raw_covmat = np.loadtxt(f) 
    
    # load probe suffix
    probe = yamlopts["train_args"]["probe"]
    #---------------------------------------------------------------------------
    # Reorder fiducial, bounds and covmat to follow ['train_args']['ord']
    #---------------------------------------------------------------------------
    # Reorder fiducial
    self.fiducial = np.array([fid[p] for p in self.sampled_params], 
                             copy=True, dtype=np.float64)

    # Reorder covmat
    pidx = {p : i for i, p in enumerate(raw_covmat_params_names)}
    try:
      idx = np.array([pidx[p] for p in self.sampled_params], copy=True, dtype=int)
    except KeyError as e:
      raise ValueError(f"{e.args[0]!r} not found in cov header") from None
    covmat = raw_covmat[np.ix_(idx, idx)]
    
    # Reorder bounds
    self.bounds = np.array(self.model.prior.bounds(confidence=0.999999),
                           copy=True, dtype=np.float64)[idx,:]    

    #---------------------------------------------------------------------------
    # Reduce correlation on the covariance matrix to max = target
    #---------------------------------------------------------------------------
    sig   = np.sqrt(np.diag(covmat))
    n = len(sig)
    outer = np.outer(sig, sig)
    corr  = covmat / outer 
    m = np.abs(corr - np.eye(n)).max()
    corr /= max(1.0, m / target) if m > 0 else 1.0
    np.fill_diagonal(corr, 1.0)
    covmat = corr * outer

    #---------------------------------------------------------------------------
    # Compute covmat inverse
    #---------------------------------------------------------------------------
    C = 0.5 * (covmat + covmat.T)  # enforce symmetry
    jitt = 0.0
    for _ in range(10):
      try:
        L = np.linalg.cholesky(C + jitt*np.eye(C.shape[0]))
        break
      except np.linalg.LinAlgError:
        scale = np.mean(np.diag(C)) # scale jitt to matrix sz start tiny -> grow
        jitt = (1e-12 if jitt==0 else jitt*10) * (scale if scale > 0 else 1.)
    else:
      raise np.linalg.LinAlgError("could not stabilized cov to SPD w/ jitter")
    I = np.eye(C.shape[0])
    self.covmat = C + jitt*np.eye(C.shape[0])
    self.inv_covmat = np.linalg.solve(L.T, np.linalg.solve(L, I))

    #---------------------------------------------------------------------------
    # Define output files
    #---------------------------------------------------------------------------
    datavsfile = Path(args.datavsfile).stem
    self.dvsf = f"{root}/chains/{datavsfile}_{probe}"
    paramfile = Path(args.paramfile).stem
    self.paramsf = f"{root}/chains/{args.paramfile}_{probe}"

  #-----------------------------------------------------------------------------
  # likelihood
  #-----------------------------------------------------------------------------
  def param_logpost(self,x):
    y = x - self.fiducial
    logprior = self.model.prior.logp(x)
    return (-0.5*(y @ self.inv_covmat @ y) + logprior)/args.temp

  #-----------------------------------------------------------------------------
  # run mcmc
  #-----------------------------------------------------------------------------
  def run_mcmc(self):
    ndim     = len(self.sampled_params)
    names    = list(self.sampled_params)
    bds      = self.bounds
    nwalkers = int(3*ndim)
    nsteps   = int(max(7500, args.nparams/nwalkers)) # (for safety we assume tau>100)
    burnin   = int(0.3*nsteps)                       # 30% burn-in
    thin     = int(float((nsteps-burnin)*nwalkers)/args.nparams)

    sampler = emcee.EnsembleSampler(nwalkers = nwalkers, 
                                    ndim = ndim, 
                                    moves=[(emcee.moves.DEMove(), 0.8),
                                           (emcee.moves.DESnookerMove(), 0.2)],
                                    log_prob_fn = self.param_logpost)
    sampler.run_mcmc(initial_state = self.fiducial[np.newaxis] + 
                                     0.5*np.sqrt(np.diag(self.covmat))*
                                     np.random.normal(size=(nwalkers, ndim)), 
                     nsteps=nsteps, 
                     progress=False)
    tau = np.array(sampler.get_autocorr_time(quiet=True, has_walkers=True),
                   copy=True, dtype=np.float64).max()
    print(f"Partial Result: tau = {tau}\n"
          f"nwalkers={nwalkers}\n"
          f"nsteps (per walker) = {nsteps}\n"
          f"nsteps/tau = {nsteps/tau} (min should be ~50)\n")
    xf   = sampler.get_chain(flat=True, discard=burnin, thin=thin)
    lnp  = sampler.get_log_prob(flat=True, discard=burnin, thin=thin)
    w    = np.ones((len(xf), 1), dtype=np.float64)
    chi2 = -2*lnp

    self.samples = xf # we just need the samples to compute the data vector

    # save chain begins --------------------------------------------------------
    hd=f"nwalkers={nwalkers}\n"
    np.savetxt(f"{self.paramsf}.1.txt",
               np.concatenate([w, lnp[:,None], xf, chi2[:,None]], axis=1),
               fmt="%.9e",
               header=hd + ''.join(["weights", "lnp"] + names),
               comments="# ")
    
    # save a range files -------------------------------------------------------
    hd = ["weights","lnp"] + names + ["chi2*"]
    rows = [(str(n),float(l),float(h)) for n,l,h in zip(names,bds[:,0],bds[:,1])]
    with open(f"{self.paramsf}.ranges", "w") as f: 
      f.write(f"# {' '.join(hd)}\n")
      f.writelines(f"{n} {l:.5e} {h:.5e}\n" for n, l, h in rows)

    # save paramname files -----------------------------------------------------
    param_info = self.model.info()['params']
    latex  = [param_info[x]['latex'] for x in names]
    names.append("chi2*")
    latex.append("\\chi^2")
    np.savetxt(f"{self.paramsf}.paramnames", 
               np.column_stack((names,latex)),
               fmt="%s")

    # save a cov matrix --------------------------------------------------------
    samples = loadMCSamples(f"{self.paramsf}", settings={'ignore_rows': u'0.0'})
    np.savetxt(f"{self.paramsf}.covmat",
               np.array(samples.cov(), copy=True, dtype=np.float64),
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
          np.save(f"{self.dvsf}.npy", self.datavectors)
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
  generator = dataset()
  if (rank == 0):
    generator.run_mcmc()
    if not args.chain:
      generator.generate_datavectors()
  else:
    if not args.chain:
      generator.generate_datavectors()
  MPI.Finalize()
  exit(0)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------