import numpy as np
import emcee, argparse, os, sys, scipy, yaml, time, traceback
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from mpi4py import MPI
from tqdm import tqdm
from pathlib import Path
from getdist import loadMCSamples
from collections import deque
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#mpirun -n 2 --oversubscribe \
#  python external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py \
#    --root projects/roman_real/  \
#    --fileroot emulators/nla_cosmic_shear/ \
#    --nparams 1000 \
#    --temp 128 \
#    --yaml 'w0wa_takahashi_cs_CNN.yaml' \
#    --datavsfile 'w0wa_takahashi_dvs_train' \
#    --paramfile 'w0wa_takahashi_params_train' \
#    --failfile  'w0wa_params_failed_train' \
#    --chain 1 \
#    --maxcorr 0.15 \
#    --loadchk 1
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
parser.add_argument("--mode",
                    dest="mode",
                    help="generation mode = [train, valid, test]",
                    type=str,
                    choices=["train","valid","test"],
                    default='train')
parser.add_argument("--chain",
                    dest="chain",
                    help="only compute and output train/test/val chain",
                    type=int,
                    choices=[0,1],
                    default=1)
parser.add_argument("--nparams",
                    dest="nparams",
                    help="Number of Parameters to Generate",
                    type=int,
                    default=100000)
parser.add_argument("--temp",
                    dest="temp",
                    help="Number of Parameters to Generate",
                    type=int,
                    default=128)
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
parser.add_argument("--maxcorr",
                    dest="maxcorr",
                    help="Max correlation allowed",
                    type=float,
                    default=0.15)
parser.add_argument("--loadchk",
                    dest="loadchk",
                    help="Load from chk if exists",
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
    self.failed = None        # track which models failed to compute dv
    self.loadchk = False      # load checkpoint if available
    self.loadsamples = False  # loaded samples sucessfully
    self.model = None
    self.sampled_params = None 
    self.fiducial = None
    self.bounds = None
    self.covmat = None 
    self.inv_covmat = None 
    self.dvsf = None 
    self.paramsf = None 
    self.failf = None
    self.setup = False
    
    self.__setup_flags()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
      if args.loadchk:
        self.__load_chk()
      if not self.loadchk:
        self.__run_mcmc()
      if not args.chain == 1:
        self.__generate_datavectors()
    else:
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
    self.failed = None        # track which models failed to compute dv
    self.loadchk = False      # load checkpoint if available
    self.loadsamples = False  # loaded samples sucessfully
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
                             copy=True, dtype=np.float32)

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
                           copy=True, dtype=np.float32)[idx,:] 
    #---------------------------------------------------------------------------
    # Reduce correlation on the covariance matrix to max = args.maxcorr
    #---------------------------------------------------------------------------
    sig   = np.sqrt(np.diag(covmat))
    n = len(sig)
    outer = np.outer(sig, sig)
    corr  = covmat / outer 
    m = np.abs(corr - np.eye(n)).max()
    corr /= max(1.0, m / args.maxcorr) if m > 0 else 1.0
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
    self.dvsf = f"{root}/chains/{datavsfile}_{probe}_{args.temp}"
    paramfile = Path(args.paramfile).stem
    self.paramsf = f"{root}/chains/{paramfile}_{probe}_{args.temp}"
    failfile = Path(args.failfile).stem
    self.failf = f"{root}/chains/{failfile}_{probe}_{args.temp}"
    
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
    return (-0.5*(y @ self.inv_covmat @ y) + logprior)/args.temp

  #-----------------------------------------------------------------------------
  # save/load checkpoint
  #-----------------------------------------------------------------------------
  def __load_chk(self):
    self.loadchk = all([os.path.isfile(x) for x in [f"{self.dvsf}.npy", 
                                                    f"{self.failf}.txt",
                                                    f"{self.paramsf}.covmat", 
                                                    f"{self.paramsf}.ranges",
                                                    f"{self.paramsf}.1.txt"]])
    if self.loadchk:
      # row 0/1 rows are weights, lnp. Last row is chi2
      self.samples = np.atleast_2d(np.loadtxt(f"{self.paramsf}.1.txt", 
                                              dtype=np.float32))[:,2:-1]
      if self.samples.ndim != 2:
        raise ValueError(f"samples must be 2D, got {self.samples.shape}") 
      
      self.failed = np.loadtxt(f"{self.failf}.txt", dtype=np.uint8)
      self.failed = np.asarray(self.failed).astype(bool)
      if self.failed.ndim != 1:
        raise ValueError(f"failed must be 1D, got {self.failed.shape}") 
      
      if self.samples.shape[0] != self.failed.shape[0]:
        raise ValueError(f"Incompatible samples/failed chk files")

      self.datavectors = np.load(f"{self.dvsf}.npy", allow_pickle=False)
      if self.datavectors.ndim != 2:
        raise ValueError(f"datavectors must be 2D, got {self.datavectors.shape}") 
      if self.datavectors.shape[0] != self.samples.shape[0]:
        raise ValueError(f"Incompatible samples/datavectir chk files")
      self.loadsamples = True
    return self.loadchk
  
  def __save_chk(self):
    np.save(f"{self.dvsf}.tmp.npy", self.datavectors)
    np.savetxt(f"{self.failf}.tmp.txt", self.failed.astype(np.uint8), fmt="%d")
    os.replace(f"{self.dvsf}.tmp.npy", f"{self.dvsf}.npy")
    os.replace(f"{self.failf}.tmp.txt", f"{self.failf}.txt")

  #-----------------------------------------------------------------------------
  # run mcmc
  #-----------------------------------------------------------------------------
  def __run_mcmc(self):
    ndim     = len(self.sampled_params)
    names    = list(self.sampled_params)
    bds      = self.bounds
    nwalkers = int(3*ndim)
    nsteps   = int(max(7500, args.nparams/nwalkers)) # (for safety we assume tau>100)
    burnin   = int(0.3*nsteps)                       # 30% burn-in
    thin     = max(1,int(float((nsteps-burnin)*nwalkers)/args.nparams)-1)
    
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
    tau = np.array(sampler.get_autocorr_time(quiet=True, has_walkers=True),
                   copy=True, dtype=np.float32).max()
    print(f"Partial Result: tau = {tau}\n"
          f"nwalkers={nwalkers}\n"
          f"nsteps (per walker) = {nsteps}\n"
          f"nsteps/tau = {nsteps/tau} (min should be ~50)\n")
    xf   = sampler.get_chain(flat = True, discard = burnin, thin = thin)
    lnp  = sampler.get_log_prob(flat = True, discard = burnin, thin = thin)
    w    = np.ones((len(xf), 1), dtype = np.float32)
    chi2 = -2*lnp

    self.samples = np.array(xf, copy=True, dtype=np.float32)
                           
    # save chain begins --------------------------------------------------------
    hd=f"nwalkers={nwalkers}\n"
    np.savetxt(f"{self.paramsf}.1.txt",
               np.concatenate([w, lnp[:,None], xf, chi2[:,None]], axis=1),
               fmt="%.9e",
               header=hd + ' '.join(["weights", "lnp"] + names),
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
    self.loadsamples = True
    return self.loadsamples
  
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
    return np.array(likelihood.get_datavector(**param), copy=True, dtype=np.float32)

  def __generate_datavectors(self):
    if not self.setup:
      raise RuntimeError(f"Initial Setup not successful")
    TASK_TAG = 1
    STOP_TAG = 0
    RESULT_TAG = 2
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    nworkers = size - 1
        
    if size == 1:
      if not self.loadsamples:
        raise RuntimeError(f"Model Samples not loaded/computed")

      nparams = len(self.samples)

      if not self.loadchk:
        self.failed = np.ones(nparams, dtype=np.uint8) # start w/ all failed
        self.failed = np.asarray(self.failed).astype(bool)
        
        try: # First run: get data vector size
          dvs = self._compute_dvs_from_sample(self.samples[0])
        except Exception:
          raise RuntimeError(f"Failed in _compute_dvs_from_sample\n" 
                             f"Cannot determine datavector length.")
        self.datavectors = np.zeros((nparams, len(dvs)), dtype=np.float32)
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

        if i % 2500 == 0:
          print(f"Model number: {i+1} (total: {nparams}) - checkpoint", flush=True)
          self.__save_chk()
      self.__save_chk()
    else:        
      if rank == 0:
        if not self.loadsamples:
          sys.stderr.write(f"Model Samples not loaded/computed\n")
          sys.stderr.flush()
          comm.Abort(1)
        status = MPI.Status() 
        block = 10000
        next_block = 1 
        too_frequent = True
        nparams = len(self.samples)
        if nparams <= nworkers:
            sys.stderr.write(f"num params (={nparams}) <= "
                             f"MPI workers (={nworkers})\n")
            sys.stderr.flush()
            comm.Abort(1)    
        completed = np.zeros(nparams, dtype=np.uint8)

        if not self.loadchk:
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
          self.datavectors = np.zeros((nparams, len(dvs)),dtype=np.float32)
          self.datavectors[0] = dvs
          self.failed[0] = False
          completed[0] = True
          idx0 = np.arange(1, nparams)
        else:
          completed = ~self.failed
          idx0 = np.where(self.failed == True)[0]
        
        tasks = deque(idx0.tolist())
        nactive = min(nworkers, len(tasks))
        for w in range(1, nactive+1):
          j = tasks.popleft() 
          comm.send((j, self.samples[j]), dest=w, tag=TASK_TAG)

        count = 0
        while tasks:
          kind, idx, dvs = comm.recv(source = MPI.ANY_SOURCE,
                                     tag = RESULT_TAG,
                                     status = status)
          if kind == "err": # set datavector to zero and continue
            self.datavectors[idx,:] = 0.0
            src = status.Get_source()
            sys.stderr.write(f"[Rank 0] Worker {src} failed at idx={idx}\n")
            sys.stderr.flush() 
            self.failed[idx] = True
          else:
            self.datavectors[idx] = dvs 
            self.failed[idx] = False
          completed[idx] = True

          if not self.loadchk:
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
          count += 1
          comm.send((j, self.samples[j]), 
                    dest = status.Get_source(), 
                    tag  = TASK_TAG)

        for _ in range(nactive): # drain active workers
          kind, idx, dvs = comm.recv(source = MPI.ANY_SOURCE, 
                                     tag = RESULT_TAG, 
                                     status = status) # drain results 
          if kind == "err":
            self.failed[idx] = True
            src = status.Get_source()
            sys.stderr.write(f"[Rank 0] Worker {src} failed at idx={idx}\n")
            sys.stderr.flush()
          else:
            self.datavectors[idx] = dvs 
            self.failed[idx] = False
          completed[idx] = True
        for w in range(1, nworkers + 1): # stop workers
          comm.send((0, None), dest=w, tag=STOP_TAG)
        comm.Barrier()  
        self.__save_chk()
      
      else:
      
        status = MPI.Status()
        while (True):
          idx, sample = comm.recv(source = 0, 
                                  tag = MPI.ANY_TAG, 
                                  status = status)
          if (status.Get_tag() == STOP_TAG):
            comm.Barrier() # if work is done (tag = 0), wait for other workers
            break
          try:
            dvs = self._compute_dvs_from_sample(sample)
            comm.send(("ok", idx, dvs), dest = 0, tag = RESULT_TAG)
          except Exception:
            comm.send(("err", idx, None), dest = 0, tag = RESULT_TAG)
            continue
    return

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  generator = dataset()
  MPI.Finalize()
  exit(0)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------