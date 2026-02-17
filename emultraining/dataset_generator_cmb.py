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
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# (...) TODO
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Command line args
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='dataset_generator_cmb')

parser.add_argument("--yaml",
                    dest="yaml",
                    help="The training YAML containing the training_args block",
                    type=str,
                    required=True)
parser.add_argument("--root",
                    dest="root",
                    help="Project folder",
                    type=str,
                    required=True)
parser.add_argument("--fileroot",
                    dest="fileroot",
                    help="Subfolder of Project folder where we find yaml and fisher",
                    type=str,
                    required=True)
parser.add_argument("--datavsfile",
                    dest="datavsfile",
                    help="File to save data vectors",
                    type=str,
                    required=True)
parser.add_argument("--paramfile",
                    dest="paramfile",
                    help="File to save parameters",
                    type=str,
                    required=True)
parser.add_argument("--failfile",
                    dest="failfile",
                    help="File that tells which cosmo param fail to compute dvs",
                    type=str,
                    required=True)
parser.add_argument("--chain",
                    dest="chain",
                    help="only compute and output train/test/val chain",
                    type=int,
                    choices=[0,1],
                    default=0)
parser.add_argument("--nparams",
                    dest="--nparams",
                    help="Requested Number of Parameters",
                    type=int)
parser.add_argument("--unif",
                    dest="unif",
                    help="Choose Between Uniform and Fisher based samples",
                    type=int,
                    choices=[0,1])
parser.add_argument("--temp",
                    dest="temp",
                    help="Number of Parameters to Generate",
                    type=int)
parser.add_argument("--maxcorr",
                    dest="maxcorr",
                    help="Max correlation allowed",
                    type=float)
parser.add_argument("--loadchk",
                    dest="loadchk",
                    help="Load from chk if exists",
                    type=int,
                    choices=[0,1])
parser.add_argument("--freqchk",
                    dest="freqchk",
                    help="Load from chk if exists",
                    type=int)
parser.add_argument("--append",
                    dest="append",
                    help="Append more models (only trye of loadchk == true)",
                    type=int,
                    choices=[0,1])
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
  def __init__(self):
    self.setup = False
    self.__setup_flags()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
      self.__run_mcmc()
    if not args.chain == 1:
      self.__generate_datavectors()

  def __setup_flags(self):
    #---------------------------------------------------------------------------
    # Basic definitions
    #---------------------------------------------------------------------------
    root_env = os.environ.get("ROOTDIR")
    if not root_env:
      raise RuntimeError("ROOTDIR environment variable is not set")
    root = root_env.rstrip("/")
    root = f"{root}/{args.root.rstrip('/')}"
    fileroot = f"{root}/{args.fileroot.rstrip('/')}"
    Path(f"{root}/chains").mkdir(parents=True, exist_ok=True)
    
    self.append = 0 if args.append is None else args.append
    self.bounds = None
    self.covmat = None 
    self.dtype = np.float32
    self.dvsf = None 
    self.dvs_is_memmap = False
    self.freqchk = 5000 if args.freqchk is None else args.freqchk
    if self.freqchk < 500:
      raise ValueError("--freqchk must be >= 500") # avoid too much chk
    self.failed = None        # track which models failed to compute dv
    self.failf = None
    self.fiducial = None
    self.inv_covmat = None
    self.loadchk = 0 if args.loadchk is None else args.loadchk 
    self.loadedfromchk = False  # check if loaded from checkpoint sucessfully
    self.loadedsamples = False  # check loaded samples sucessfully
    self.maxcorr = 0.2 if args.maxcorr is None else args.maxcorr
    if not (0.01 < self.maxcorr <= 1):
      raise ValueError("--maxcorr must be between (0.01,1]")
    self.model = None
    self.nparams = 10000 if args.nparams is None else args.nparams
    if self.nparams < 0:
      raise ValueError("--nparams must be positive integer")
    self.paramsf = None 
    self.sampled_params = None 
    self.samples = None
    self.temp = 128 if args.temp is None else args.temp
    if self.temp < 1:
      raise ValueError("--temp must be > 1")
    self.unif = 1 if args.unif is None else args.unif
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

    if not self.unif == 1:
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
                               copy=True, dtype=self.dtype)

      # Reorder covmat
      pidx = {p : i for i, p in enumerate(raw_covmat_params_names)}
      try:
        idx = np.array([pidx[p] for p in self.sampled_params], copy=True, dtype=int)
      except KeyError as e:
        raise ValueError(f"{e.args[0]!r} not found in cov header") from None
      covmat = raw_covmat[np.ix_(idx, idx)]
    
    # Reorder bounds
    names = list(self.model.parameterization.sampled_params().keys())
    pidx = {p : i for i, p in enumerate(names)}
    idx = np.array([pidx[p] for p in self.sampled_params], copy=True, dtype=int)
    self.bounds = np.array(self.model.prior.bounds(confidence=0.999999),
                           copy=True, dtype=self.dtype)[idx,:] 
    
    if not self.unif == 1:
      #-------------------------------------------------------------------------
      # Reduce correlation on the covariance matrix to max = args.maxcorr
      #-------------------------------------------------------------------------
      sig   = np.sqrt(np.diag(covmat))
      n = len(sig)
      outer = np.outer(sig, sig)
      corr  = covmat / outer 
      m = np.abs(corr - np.eye(n)).max()
      corr /= max(1.0, m / self.maxcorr) if m > 0 else 1.0
      np.fill_diagonal(corr, 1.0)
      covmat = corr * outer

      #-------------------------------------------------------------------------
      # Compute covmat inverse
      #-------------------------------------------------------------------------
      C = 0.5 * (covmat + covmat.T)  # enforce symmetry
      jitt = 0.0
      for _ in range(10):
        try:
          L = np.linalg.cholesky(C + jitt*np.eye(C.shape[0]))
          break
        except np.linalg.LinAlgError:
          scale = np.mean(np.diag(C)) # scale jitt to matrix sz start tiny -> grow
          jitt = (1e-12 * scale if jitt == 0 else jitt*10)
      else:
        raise np.linalg.LinAlgError("could not stabilized cov to SPD w/ jitter")
      I = np.eye(C.shape[0])
      self.covmat = C + jitt*np.eye(C.shape[0])
      self.inv_covmat = np.linalg.solve(L.T, np.linalg.solve(L, I))

    #---------------------------------------------------------------------------
    # Define output files
    #---------------------------------------------------------------------------
    datavsfile = Path(args.datavsfile).stem
    self.dvsf = f"{root}/chains/{datavsfile}_{probe}_{self.temp}"
    paramfile = Path(args.paramfile).stem
    self.paramsf = f"{root}/chains/{paramfile}_{probe}_{self.temp}"
    failfile = Path(args.failfile).stem
    self.failf = f"{root}/chains/{failfile}_{probe}_{self.temp}"
    
    #---------------------------------------------------------------------------
    # Setup Done
    #---------------------------------------------------------------------------
    self.setup = True

  #-----------------------------------------------------------------------------
  # likelihood
  #-----------------------------------------------------------------------------
  def __param_logpost(self,x):
    y = x - self.fiducial
    logprior = self.model.prior.logp(x)
    return (-0.5*(y @ self.inv_covmat @ y) + logprior)/self.temp

  #-----------------------------------------------------------------------------
  # save/load checkpoint
  #-----------------------------------------------------------------------------
  def __load_chk(self):
    rtnvar = False
    if self.loadchk == 1:
      loadchk = all([os.path.isfile(x) for x in [f"{self.dvsf}.npy", 
                                                 f"{self.failf}.txt",
                                                 f"{self.paramsf}.covmat", 
                                                 f"{self.paramsf}.ranges",
                                                 f"{self.paramsf}.1.txt"]])
      if loadchk:
        # row 0/1 rows are weights, lnp. Last row is chi2
        self.samples = np.atleast_2d(np.loadtxt(f"{self.paramsf}.1.txt", 
                                                dtype=self.dtype))[:,2:-1]
        if self.samples.ndim != 2:
          raise ValueError(f"samples must be 2D, got {self.samples.shape}") 
        
        self.failed = np.loadtxt(f"{self.failf}.txt", dtype=np.uint8)
        self.failed = np.asarray(self.failed).astype(bool)
        if self.failed.ndim != 1:
          raise ValueError(f"failed must be 1D, got {self.failed.shape}") 
        
        if self.samples.shape[0] != self.failed.shape[0]:
          print(self.samples.shape[0], self.failed.shape[0])
          raise ValueError(f"Incompatible samples/failed chk files")

        # load datavectors begins ----------------------------------------------
        arr = np.load(f"{self.dvsf}.npy", 
                      mmap_mode="r", 
                      allow_pickle=False)
        RAMneed = arr.nbytes + self.samples.nbytes + self.failed.nbytes
        RAMavail = psutil.virtual_memory().available
        if RAMneed < 0.75 * RAMavail:
          self.datavectors = np.load(f"{self.dvsf}.npy", allow_pickle=False)
          self.dvs_is_memmap = False
        else:
          print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM."
                f"There is {RAMavail/1e9:.2f} GB of RAM available."
                f"We will read dvs from HD (slow)")
          self.datavectors = np.load(f"{self.dvsf}.npy", 
                                     mmap_mode="r+", 
                                     allow_pickle=False)
          self.dvs_is_memmap = True
        del arr
        # load datavectors ends ------------------------------------------------

        if self.datavectors.ndim != 2:
          raise ValueError(f"datavectors must be 2D, got {self.datavectors.shape}") 
        if self.datavectors.shape[0] != self.samples.shape[0]:
          raise ValueError(f"Incompatible samples/datavectir chk files")
        if loadchk: 
          print("Loaded models from chk")
          if self.append == 0:
            self.loadedsamples = True
            self.loadedfromchk = True
        rtnvar = True
    return rtnvar
  
  def __save_chk(self):
    if self.dvs_is_memmap == True:
      self.datavectors.flush()  # checkpoint dv in-place
    else:
      np.save(f"{self.dvsf}.tmp.npy", self.datavectors)
      os.replace(f"{self.dvsf}.tmp.npy", f"{self.dvsf}.npy")
    np.savetxt(f"{self.failf}.tmp.txt", self.failed.astype(np.uint8), fmt="%d")
    os.replace(f"{self.failf}.tmp.txt", f"{self.failf}.txt")

  #-----------------------------------------------------------------------------
  # run mcmc
  #-----------------------------------------------------------------------------
  def __run_mcmc(self):
    try:
      loadedfromchk = self.__load_chk()
    except Exception as e:
      sys.stderr.write(f"[load_chk] failed: {e}\n")
      traceback.print_exc(file=sys.stderr)
      loadedfromchk = False
    
    if (loadedfromchk == False) or (loadedfromchk == True and self.append == 1):
      ndim     = len(self.sampled_params)
      names    = list(self.sampled_params)
      bds      = self.bounds
      
      if not self.unif == 1
        nwalkers = int(3*ndim)
        nsteps   = int(max(7500, self.nparams/nwalkers)) # (for safety we assume tau>100)
        burnin   = int(0.3*nsteps)                       # 30% burn-in
        thin     = max(1,int(float((nsteps-burnin)*nwalkers)/self.nparams)-1)
        
        sampler = emcee.EnsembleSampler(nwalkers = nwalkers, 
                                        ndim = ndim, 
                                        moves=[(emcee.moves.DEMove(), 0.8),
                                               (emcee.moves.DESnookerMove(), 0.2)],
                                        log_prob_fn = self.__param_logpost)
        sampler.run_mcmc(initial_state = self.fiducial[np.newaxis] + 
                                         0.5*np.sqrt(np.diag(self.covmat))*
                                         np.random.normal(size=(nwalkers, ndim)), 
                         nsteps=nsteps, 
                         progress=False)

        xf   = sampler.get_chain(flat = True, discard = burnin, thin = thin)
        nparams = np.atleast_2d(xf).shape[0]
        lnp  = sampler.get_log_prob(flat = True, discard = burnin, thin = thin)
      else:
        # TODO
        lnp  = np.ones((nparams, 1), dtype=np.uint8)
      
      w    = np.ones((nparams, 1), dtype=np.uint8)
      chi2 = -2*lnp
      
      if not loadedfromchk:
        # Output some debug messaging ------------------------------------------
        try:
          tau = np.array(sampler.get_autocorr_time(quiet=True, has_walkers=True),
                       copy=True, 
                       dtype=self.dtype).max()
          print(f"Partial Result: tau = {tau}\n"
                f"nwalkers={nwalkers}\n"
                f"nsteps (per walker) = {nsteps}\n"
                f"nsteps/tau = {nsteps/tau} (min should be ~50)\n"
                f"nparams (after thin)={nparams}\n")
        except Exception as e:
          print(f"Partial Result: tau = N/A (emcee threw an exception)\n"
                f"nwalkers={nwalkers}\n"
                f"nsteps (per walker) = {nsteps}\n"
                f"nparams (after thin)={nparams}\n")
          tau = 1 # make sure main MPI worker does not crash over trivial check
        # save a range files ---------------------------------------------------
        hd = ["weights","lnp"] + names + ["chi2*"]
        rows = [(str(n),float(l),float(h)) for n,l,h in zip(names,bds[:,0],bds[:,1])]
        with open(f"{self.paramsf}.ranges", "w") as f: 
          f.write(f"# {' '.join(hd)}\n")
          f.writelines(f"{n} {l:.5e} {h:.5e}\n" for n, l, h in rows)
        # save paramname files -------------------------------------------------
        param_info = self.model.info()['params']
        latex  = [param_info[x]['latex'] for x in names]
        names.append("chi2*")
        latex.append("\\chi^2")
        np.savetxt(f"{self.paramsf}.paramnames", 
                   np.column_stack((names,latex)),
                   fmt="%s")
        # save chain begins ----------------------------------------------------
        fname = f"{self.paramsf}.1.txt";
        hd=f"nwalkers={nwalkers}\n"
        np.savetxt(fname,
                   np.concatenate([w, lnp[:,None], xf, chi2[:,None]], axis=1),
                   fmt="%.9e",
                   header=hd + ' '.join(["weights", "lnp"] + names),
                   comments="# ")
        # copy samples to self.samples  ----------------------------------------
        self.samples = np.array(xf, copy=True, dtype=self.dtype)
        del w         # save RAM memory
        del xf        # save RAM memory
        del lnp       # save RAM memory
        del chi2      # save RAM memory
        gc.collect()  # save RAM memory
      else:
        # append chain file begins ---------------------------------------------
        fname = f"{self.paramsf}.1.txt";
        with open(fname, "a") as f: # append mode
          hd = f"nwalkers={nwalkers}\n" + ' '.join(["weights","lnp"]+names)
          np.savetxt(f, 
                     np.concatenate([w, lnp[:,None], xf, chi2[:,None]], axis=1), 
                     header = hd if (os.path.getsize(fname) == 0) else "",
                     fmt = "%.9e")
        del w         # save RAM memory
        del xf        # save RAM memory
        del lnp       # save RAM memory
        del chi2      # save RAM memory
        gc.collect()  # save RAM memory
        
        self.samples = np.atleast_2d(np.loadtxt(fname, dtype=self.dtype))[:,2:-1]
        if self.samples.ndim != 2:
          raise ValueError(f"samples must be 2D, got {self.samples.shape}") 

        # append fail file begins ----------------------------------------------
        fname = f"{self.failf}.txt";
        failed = np.ones((nparams, 1), dtype=np.uint8) # start w/ all failed
        with open(fname, "a") as f: # append mode
          np.savetxt(f, failed.astype(np.uint8), fmt="%d")

        self.failed = np.loadtxt(fname, dtype=np.uint8)
        self.failed = np.asarray(self.failed).astype(bool)
        if self.failed.ndim != 1:
          raise ValueError(f"failed must be 1D, got {self.failed.shape}") 

        if self.samples.shape[0] != self.failed.shape[0]:
          raise ValueError(f"Incompatible samples/failed chk files")

        # Expand dvs begins ----------------------------------------------------
        nrows = self.datavectors.shape[0]
        ncols = self.datavectors.shape[1]
 
        RAMneed = ( self.samples.nbytes + self.failed.nbytes + 
                    self.datavectors.nbytes + 
                    (nrows + nparams)*ncols*self.datavectors.dtype.itemsize)
        RAMavail = psutil.virtual_memory().available
        if RAMneed < 0.75 * RAMavail:
          self.datavectors = np.vstack((self.datavectors, 
                                        np.zeros((nparams,ncols),dtype=self.dtype)))
          np.save(f"{self.dvsf}.tmp.npy", self.datavectors)
          os.replace(f"{self.dvsf}.tmp.npy", f"{self.dvsf}.npy")
          self.dvs_is_memmap = False
        else:
          print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM."
                f"There is {RAMavail/1e9:.2f} GB of RAM available."
                f"We will read dvs from HD (slow)")
          datavectors = open_memmap(f"{self.dvsf}.tmp.npy", 
                                    mode="w+",
                                    shape=(nrows + nparams, ncols),
                                    dtype=self.datavectors.dtype)
          for s in range(0, nrows, 2500): # chunks = avoid RAM spikes) 
            e = min(nrows, s + 2500)
            datavectors[s:e] = self.datavectors[s:e]
          for s in range(nrows, nrows + nparams, 2500):
            e = min(nrows + nparams, s + 2500)
            datavectors[s:e] = 0
          datavectors.flush()
          del datavectors
          os.replace(f"{self.dvsf}.tmp.npy", f"{self.dvsf}.npy")
          self.datavectors = np.load(f"{self.dvsf}.npy", 
                                     mmap_mode="r+", 
                                     allow_pickle=False)
          self.dvs_is_memmap = True
        # Expand dvs ends ------------------------------------------------------
        
        # check final dimensions -----------------------------------------------
        if self.datavectors.shape[0] != self.samples.shape[0]:
          raise ValueError(f"Incompatible samples/datavectir chk files")
        
        # set  self.loadedfromchk ----------------------------------------------
        self.loadedfromchk = True

      # save a cov matrix ------------------------------------------------------
      samples = loadMCSamples(f"{self.paramsf}", settings={'ignore_rows': u'0.'})
      np.savetxt(f"{self.paramsf}.covmat",
                 np.array(samples.cov(), copy=True, dtype=self.dtype),
                 fmt="%.9e",
                 header=' '.join(names),
                 comments="# ")
    # set self.loadedsamples ---------------------------------------------------
    self.loadedsamples = True
  
  #-----------------------------------------------------------------------------
  # datavectors
  #-----------------------------------------------------------------------------
  def _compute_dvs_from_sample(self, sample):
    param = self.model.parameterization.to_input(
        sampled_params_values=dict(zip(self.sampled_params, sample))
    )
    self.model.provider.set_current_input_params(param)
    
    likelihood = self.model.likelihood[list(self.model.likelihood.keys())[0]]

    for (x, _), z in zip(self.model._component_order.items(),
                         self.model._params_of_dependencies):
        x.check_cache_and_compute(
            params_values_dict={p: param[p] for p in x.input_params},
            want_derived={},
            dependency_params=[param[p] for p in z],
            cached=False
        )
    return np.array(likelihood.get_datavector(**param), 
                    copy=True, 
                    dtype=self.dtype)

  def __generate_datavectors(self):
    if not self.setup:
      raise RuntimeError(f"Initial Setup not successful")
    TTAG = 1      # Task tag
    STAG = 0      # Stop tag
    RTAG = 2      # Result tag
    DTAG = 3      # Done (not crashed) tag
    TASK_TIMEOUT = 1800
    STOP_TIMEOUT = 300.0 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    nworkers = size - 1
        
    if size == 1:
      if not self.loadedsamples:
        raise RuntimeError(f"Model Samples not loaded/computed")

      nparams = len(self.samples)

      if not self.loadedfromchk:
        self.failed = np.ones(nparams, dtype=np.uint8) # start w/ all failed
        self.failed = np.asarray(self.failed).astype(bool)
        
        try: # First run: get data vector size
          dvs = self._compute_dvs_from_sample(self.samples[0])
        except Exception:
          raise RuntimeError(f"Failed in _compute_dvs_from_sample\n" 
                             f"Cannot determine datavector length.")
        nrows = nparams
        ncols = len(dvs)
        # Allocate dvs begins --------------------------------------------------
        RAMneed = ( self.samples.nbytes + self.failed.nbytes + 
                    nrows*ncols*dvs.dtype.itemsize)
        RAMavail = psutil.virtual_memory().available
        if RAMneed < 0.75 * RAMavail:
          self.datavectors = np.zeros((nrows, ncols), dtype=self.dtype)
          self.dvs_is_memmap = False
        else:
          print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM."
                f"There is {RAMavail/1e9:.2f} GB of RAM available."
                f"We will read dvs from HD (slow)")
          self.datavectors = open_memmap(f"{self.dvsf}.npy", 
                                       mode="w+",
                                       shape=(nrows, ncols),
                                       dtype=self.dtype)
          self.datavectors[:] = 0.0
          self.datavectors.flush()
          self.dvs_is_memmap = True
        # Allocate dvs end -----------------------------------------------------
        self.datavectors[0] = dvs
        self.failed[0] = False
        idx = np.arange(1, nparams)
      else:
        idx = np.where(self.failed == True)[0] 

      for i in idx:
        try:
          dvs = self._compute_dvs_from_sample(self.samples[i])
          self.failed[i] = False
        except Exception: # set datavector to zero and continue
          self.failed[i] = True
          self.datavectors[i,:] = 0.0
          sys.stderr.write(f"[Rank 0] Worker 0 failed at i={i}\n")
          sys.stderr.flush()
          continue
        self.datavectors[i] = dvs

        if i % self.freqchk == 0:
          print(f"Model number: {i+1} (total: {nparams}) - checkpoint", flush=True)
          self.__save_chk()
      self.__save_chk()
    else:        
      if rank == 0:
        if not self.loadedsamples:
          sys.stderr.write(f"Model Samples not loaded/computed\n")
          sys.stderr.flush()
          comm.Abort(1)
        status = MPI.Status() 
        block = self.freqchk
        next_block = 1 
        too_frequent = True
        nparams = len(self.samples)   
        completed = np.zeros(nparams, dtype=np.uint8)

        if not self.loadedfromchk:
          self.failed = np.ones(nparams, dtype=np.uint8) # start w/ all failed
          self.failed = np.asarray(self.failed).astype(bool)

          try: # First run: get data vector size
            dvs = self._compute_dvs_from_sample(self.samples[0])
          except Exception:
            sys.stderr.write(f"Failed in _compute_dvs_from_sample for idx=0\n" 
                             f"Cannot determine datavector length\n"
                             f"aborting MPI job\n")
            sys.stderr.flush()
            comm.Abort(1)
          nrows = nparams
          ncols = len(dvs)
          # Allocate dvs begins ------------------------------------------------
          RAMneed = ( self.samples.nbytes + self.failed.nbytes + 
                      nrows*ncols*dvs.dtype.itemsize)
          RAMavail = psutil.virtual_memory().available
          if RAMneed < 0.75 * RAMavail:
            self.datavectors = np.zeros((nrows, ncols), dtype=self.dtype)
            self.dvs_is_memmap = False
          else:
            print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM."
                  f"There is {RAMavail/1e9:.2f} GB of RAM available."
                  f"We will read dvs from HD (slow)")
            self.datavectors = open_memmap(f"{self.dvsf}.npy", 
                                         mode="w+",
                                         shape=(nrows, ncols),
                                         dtype=self.dtype)
            self.datavectors[:] = 0.0
            self.datavectors.flush()
            self.dvs_is_memmap = True
          # Allocate dvs end ---------------------------------------------------
          self.datavectors[0] = dvs
          self.failed[0] = False
          completed[0] = True
          idx0 = np.arange(1, nparams)
        else:
          completed = ~self.failed
          idx0 = np.where(self.failed == True)[0]
        
        tasks   = deque(idx0.tolist())
        nactive = min(nworkers, len(tasks))
        active  = {} # Dict: key = worker (src), value: (idx, t_start)
        for w in range(1, nactive+1):
          j = tasks.popleft() 
          comm.send((j, self.samples[j]), dest = w, tag = TTAG)
          active[w] = (j, MPI.Wtime())

        count  = 0
        while tasks:
          # comm.Iprobe = non-blocking operation used to check for an incoming 
          #               message without actually receiving it
          # Why? protect the script against crashes (like CAMB/Class crash)
          if comm.Iprobe(source = MPI.ANY_SOURCE, tag = RTAG, status = status):
            kind, idx, dvs = comm.recv(source = MPI.ANY_SOURCE,
                                       tag = RTAG,
                                       status = status)
            count += 1
            src = status.Get_source()
            if src in active:
              del active[src]
            
            if kind == "err": # set datavector to zero and continue
              self.datavectors[idx,:] = 0.0
              self.failed[idx] = True
              sys.stderr.write(f"[Rank 0] Worker {src} failed at idx={idx}\n"
                               f"Reason: {dvs}\n")
              sys.stderr.flush() 
            else:
              self.datavectors[idx] = dvs 
              self.failed[idx] = False
            completed[idx] = True

            if not self.loadedfromchk:
              if count%block == 0:
                too_frequent = False

              if not too_frequent:
                start = (next_block - 1) * block
                end   = min(nparams, next_block * block)
                if completed[start:end].all():
                  self.__save_chk()
                  too_frequent = True
                  next_block += 1
            else:
              if count%block == 0:
                self.__save_chk()

            j = tasks.popleft() 
            comm.send((j, self.samples[j]), 
                      dest = src, 
                      tag  = TTAG)
            active[src] = (j, MPI.Wtime())
          else:
            doabort = False
            for w, (idx, t0) in list(active.items()):
              if (MPI.Wtime()-t0) > TASK_TIMEOUT: # no task runtime > TIMEOUT
                sys.stderr.write(f"[Rank 0] Worker {w} at idx={idx} timed out (MPI RTAG)")
                sys.stderr.flush()
                doabort = True
            if doabort:
              for w, (idx, t0) in list(active.items()): # mark all running tasks as failed
                self.datavectors[idx,:] = 0.0
                self.failed[idx] = True
                completed[idx] = True
              self.__save_chk() # save before crashing
              comm.Abort(1) 
            time.sleep(.1) # avoid 100% CPU usage
        # end of while loop  

        # drain active workers ------------------------------------------------
        while active: 
          # comm.Iprobe = non-blocking operation used to check for an incoming 
          #               message without actually receiving it
          # Why? protect the script against crashes (like CAMB/Class crash)
          if comm.Iprobe(source = MPI.ANY_SOURCE, tag = RTAG, status = status):
            kind, idx, dvs = comm.recv(source = MPI.ANY_SOURCE, 
                                       tag = RTAG, 
                                       status = status) # drain results 
            src = status.Get_source()
            if kind == "err":
              self.datavectors[idx,:] = 0.0
              self.failed[idx] = True
              sys.stderr.write(f"[Rank 0] Worker {src} failed at idx={idx}\n"
                               f"Reason: {dvs}\n")
              sys.stderr.flush()
            else:
              self.datavectors[idx] = dvs 
              self.failed[idx] = False
            completed[idx] = True
            if src in active: # Remove worker from active list
              del active[src]
          else:
            doabort = False
            for w, (idx, t0) in list(active.items()):
              if (MPI.Wtime()-t0) > TASK_TIMEOUT: # no task runtime > TIMEOUT
                sys.stderr.write(f"[Rank 0] Worker {w} at idx={idx} timed out (MPI RTAG)")
                sys.stderr.flush()
                doabort = True
            if doabort:
              for w, (idx, t0) in list(active.items()): # mark all running tasks as failed
                self.datavectors[idx,:] = 0.0
                self.failed[idx] = True
                completed[idx] = True
              self.__save_chk() # save before crashing
              comm.Abort(1)
            time.sleep(.1) # avoid 100% CPU usage    
        # end active workers
        
        # stop workers ---------------------------------------------------------
        self.__save_chk() # save before sending stop sign 
        active = {}       # reinitialize active (extra safety)
        for w in range(1, nworkers + 1): # stop workers
          comm.send((0, None), dest=w, tag=STAG)
          active[w] = (0, MPI.Wtime())     
        while active:
          # comm.Iprobe = non-blocking operation used to check for an incoming 
          #               message without actually receiving it
          # Why? protect the script against crashes (like CAMB/Class crash)
          if comm.Iprobe(source=MPI.ANY_SOURCE, tag=DTAG, status=status):
            _ = comm.recv(source=MPI.ANY_SOURCE, tag=DTAG, status=status)
            src = status.Get_source()
            if src in active:
              del active[src]
          else:
            for w, (_, t0) in list(active.items()):
              if (MPI.Wtime()-t0) > STOP_TIMEOUT: # no task runtime > TIMEOUT
                sys.stderr.write(f"[Rank 0] Worker {w} timed out (MPI DTAG)")
                sys.stderr.flush()
                comm.Abort(1)
            time.sleep(.1) # avoid 100% CPU usage
        # end stop workers
      
      else:
      
        status = MPI.Status()
        while (True):
          idx, sample = comm.recv(source = 0, 
                                  tag = MPI.ANY_TAG, 
                                  status = status) # try block on main b/c if
                                                   # rank zero throws an exception
                                                   # before send, MPI hangs
          if (status.Get_tag() == STAG):
            comm.send(("worker done", rank), dest = 0, tag = DTAG)
            break
          try:
            dvs = self._compute_dvs_from_sample(sample)
            comm.send(("ok", idx, dvs), dest = 0, tag = RTAG)
          except Exception:
            comm.send(("err", idx, traceback.format_exc(limit=8)), 
                      dest = 0, 
                      tag = RTAG)
            continue
    return




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
    datavectors_file_path = PATH + args.datavsfile
    parameters_file  = PATH + args.paramfile

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
            result_cls[i:total_num_dvs:num_ranks] = comm.recv(source = i, tag = 10)
            result_extra[i:total_num_dvs:num_ranks]   = comm.recv(source = i, tag = 12)

        np.save(CMB_DIR, result_cls)
        np.save(EXTRA_DIR, result_extra)
            
    else:    
        comm.send(total_cls, dest = 0, tag = 10)
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
