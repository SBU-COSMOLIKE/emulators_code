import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
import os
import sys
from datetime import datetime
import h5py as h5

sys.path.append(os.path.dirname(__file__))
from config import cocoa_config

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class Better_ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(Better_ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()#torch.nn.BatchNorm1d(in_size)
        self.norm3 = Affine()#torch.nn.BatchNorm1d(in_size)

        self.act1 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#
        self.act3 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.act1(self.norm1(self.layer1(x)))
        o2 = self.layer2(o1) + xskip             #(self.norm2(self.layer2(o1))) + xskip
        o3 = self.act3(self.norm3(o2))

        return o3

class Better_Attention(nn.Module):
    def __init__(self, in_size ,n_partitions, dropout=False):
        super(Better_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)

        self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout(p=0.1)
        else:
            self.drop = nn.Identity()

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        Q = self.WQ(_x) # query with q_i as rows
        K = self.WK(_x) # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows

        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        out = self.drop(torch.reshape(prod,(batch_size,-1)))+x # reshape back to vector

        return out

class Better_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions, dropout=False):
        super(Better_Transformer, self).__init__()  
    
        # get/set up hyperparams
        self.in_size      = in_size
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = activation_fcn(in_size)  #nn.Tanh()   #nn.ReLU()#
        self.norm         = torch.nn.BatchNorm1d(in_size)
        #self.act2         = nn.Tanh()#nn.ReLU()#
        #self.norm2        = torch.nn.BatchNorm1d(in_size)
        self.act3         = activation_fcn(in_size)  #nn.Tanh()
        self.norm3        = torch.nn.BatchNorm1d(in_size)

        # set up weight matrices and bias vectors
        weights1 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights1 = nn.Parameter(weights1) # turn the weights tensor into trainable weights
        bias1 = torch.Tensor(in_size)
        self.bias1 = nn.Parameter(bias1) # turn bias tensor into trainable weights

        weights2 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights2 = nn.Parameter(weights2) # turn the weights tensor into trainable weights
        bias2 = torch.Tensor(in_size)
        self.bias2 = nn.Parameter(bias2) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights1, a=np.sqrt(5)) # matrix weights init 
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1) # fan_in in the input size, fan out is the output size but it is not use here
        bound1 = 1 / np.sqrt(fan_in1) 
        nn.init.uniform_(self.bias1, -bound1, bound1) # bias weights init

        nn.init.kaiming_uniform_(self.weights2, a=np.sqrt(5))  
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weights2)
        bound2 = 1 / np.sqrt(fan_in2) 
        nn.init.uniform_(self.bias2, -bound2, bound2)

        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout(p=0.1)
        else:
            self.drop = nn.Identity()

    def forward(self,x):
        mat1 = torch.block_diag(*self.weights1) # how can I do this on init rather than on each forward pass?
        mat2 = torch.block_diag(*self.weights2)

        o1 = self.norm(torch.matmul(x,mat1)+self.bias1)
        o2 = self.act(o1)
        o3 = self.drop(torch.matmul(o1,mat2) + self.bias2) + x
        o4 = self.act3(o3)
        return o4

class activation_fcn(nn.Module):
    def __init__(self, dim):
        super(activation_fcn, self).__init__()

        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros((dim)))
        self.beta = nn.Parameter(torch.zeros((dim)))

    def forward(self,x):
        exp = torch.mul(self.beta,x)
        inv = torch.special.expit(exp)
        fac_2 = 1-self.gamma
        out = torch.mul(self.gamma + torch.mul(inv,fac_2), x)
        return out

class nn_emulator:
    def __init__(self, preset=None, model=None):

        # on init, we will only create the sequential model based on a preset names
        # available presets are:
        #       xi_restrf
        #       3x2_restrf
        # planned presets are:
        #       xi_cnn
        #       3x2_cnn

        layers = []

        if ( preset is None and model is None ):
            raise Exception('No preset or model was provided.')

        elif (preset is not None and model is not None ):
            raise Exception('Both a preset and a model were provide.\nOnly provide one or the other, not both!')

        elif ( preset is None and model is not None ):
            self.model = model

        elif ( preset is not None and model is None):
            if ( preset == 'xi_restrf' ):
                self.start = 0
                self.stop  = 780
                layers.append(nn.Linear(12, 256))
                layers.append(Better_ResBlock(256, 256))
                layers.append(Better_ResBlock(256, 256))
                layers.append(Better_ResBlock(256, 256))
                layers.append(nn.Linear(256, 1024))
                layers.append(Better_Attention(1024, 32))
                layers.append(Better_Transformer(1024, 32))
                layers.append(Better_Attention(1024, 32))
                layers.append(Better_Transformer(1024, 32))
                layers.append(Better_Attention(1024, 32))
                layers.append(Better_Transformer(1024, 32))
                layers.append(nn.Linear(1024,780))
                layers.append(Affine())

            elif ( preset == '3x2_restrf' ):
                self.start = 0 
                self.stop  = 1560
                layers.append(nn.Linear(22, 512))
                layers.append(Better_ResBlock(512, 512))
                layers.append(Better_ResBlock(512, 512))
                layers.append(Better_ResBlock(512, 512))
                layers.append(nn.Linear(512, 3840))
                layers.append(Better_Attention(3840, 60))
                layers.append(Better_Transformer(3840, 60))
                layers.append(Better_Attention(3840, 60))
                layers.append(Better_Transformer(3840, 60))
                layers.append(Better_Attention(3840, 60))
                layers.append(Better_Transformer(3840, 60))
                layers.append(nn.Linear(3840,1560))
                layers.append(Affine())

            else:
                raise Exception('Preset is not known! Available presets are ("xi_restrf","3x2_restrf")')

        self.model = nn.Sequential(*layers)
        self.trained = False

    def update_progress(self, train_loss, valid_loss, start_time, epoch, total_epochs, optim):
        elapsed_time = int((datetime.now() - start_time).total_seconds())
        lr = optim.param_groups[0]['lr']
        epoch=epoch+1

        width = 20
        factor = int( width * (epoch/total_epochs) )
        bar = '['
        for i in range(width):
            if i < factor:
                bar += '#'
            else:
                bar += ' '
        bar += ']'

        remaining_time = int((elapsed_time / (epoch)) * (total_epochs - (epoch)))

        print('\r' + bar + ' ' +                                \
              f'Epoch {epoch:3d}/{total_epochs:3d} | ' +        \
              f'loss={train_loss:1.3e}({valid_loss:1.3e}) | ' + \
              f'lr={lr:1.2e} | ' +                              \
              f'time elapsed={elapsed_time:7d} s; time remaining={remaining_time:7d} s',end='')

    def train(self, device, config_file,
            train_samples_file, train_datavectors_file,
            valid_samples_file, valid_datavectors_file,
            n_epochs=150, batch_size=1024, learning_rate=1e-3, reduce_lr=True, weight_decay=0,
            save_losses=False):

        # open the config file to get the data covariance.
        config = cocoa_config(config_file)

        print('Loading and processing the data. May take some time...')
        covmat = torch.as_tensor(config.cov[self.start:self.stop,self.start:self.stop],dtype=torch.float64)
        self.dv_fid  = torch.as_tensor(config.dv_fid[self.start:self.stop],dtype=torch.float64)

        # load the data of the given train_prefix and valid_prefix. Leave on cpu to save vram!
        x_train = torch.as_tensor(np.load(train_samples_file),dtype=torch.float64)
        y_train = torch.as_tensor(np.load(train_datavectors_file)[:,self.start:self.stop],dtype=torch.float64)

        x_valid = torch.as_tensor(np.load(valid_samples_file),dtype=torch.float64)
        y_valid = torch.as_tensor(np.load(valid_datavectors_file)[:,self.start:self.stop],dtype=torch.float64)

        # normalize the input parameters
        self.samples_mean = torch.Tensor(x_train.mean(axis=0, keepdims=True))
        self.samples_std  = torch.Tensor(x_train.std(axis=0, keepdims=True))

        x_train = torch.div( (x_train - self.samples_mean), self.samples_std)
        x_valid = torch.div( (x_valid - self.samples_mean), self.samples_std)

        # diagonalize the training datavectors
        self.dv_evals, self.dv_evecs = torch.linalg.eigh(covmat)

        y_train = torch.div( (y_train - self.dv_fid) @ self.dv_evecs, torch.sqrt(self.dv_evals))
        y_valid = torch.div( (y_valid - self.dv_fid) @ self.dv_evecs, torch.sqrt(self.dv_evals))

        # convert to float32
        x_train = torch.as_tensor(x_train,dtype=torch.float32)
        y_train = torch.as_tensor(y_train,dtype=torch.float32)
        x_valid = torch.as_tensor(x_valid,dtype=torch.float32)
        y_valid = torch.as_tensor(y_valid,dtype=torch.float32)

        #cuda_dv_evals = 1/self.dv_evals.to(device,dtype=torch.float32)

        # setup ADAM optimizer and reduce_lr scheduler
        optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min',patience=15,factor=0.1)
         
        # load the data into loaders
        self.model.to(device)

        generator = torch.Generator(device=device)
        trainset    = torch.utils.data.TensorDataset(x_train, y_train)
        validset    = torch.utils.data.TensorDataset(x_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)#, generator=generator)
        validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)#, generator=generator)

        # begin training
        print('Begin training...',end='')
        train_start_time = datetime.now()

        losses_train = []
        losses_valid = []
        loss = 100.

        for e in range(n_epochs):
            self.model.train()

            # training loss
            losses = []
            for i, data in enumerate(trainloader):    
                X       = data[0].to(device)
                Y_batch = data[1].to(device)
                Y_pred  = self.model(X)

                # PCA part
                diff = Y_batch - Y_pred
                chi2 = torch.diag(diff @ torch.t(diff))

                # loss = torch.mean(chi2)                      # ordinary chi2
                loss = torch.mean((1+2*chi2)**(1/2))-1       # hyperbola
                # loss = torch.mean(torch.mean(chi2**(1/2)))   # sqrt(chi2)

                losses.append(loss.cpu().detach().numpy())

                optim.zero_grad()
                loss.backward()
                optim.step()

            losses_train.append(np.mean(losses))

            ###validation loss
            losses=[]
            with torch.no_grad():
                self.model.eval()
                losses = []
                for i, data in enumerate(validloader):  
                    X_v       = data[0].to(device)
                    Y_v_batch = data[1].to(device)
                    Y_v_pred = self.model(X_v)

                    diff_v = Y_v_batch - Y_v_pred
                    chi2_v = torch.diag(diff_v @ torch.t(diff_v))

                    # loss_vali = torch.mean(chi2_v)                      # ordinary chi2
                    loss_vali = torch.mean((1+2*chi2_v)**(1/2))-1       # hyperbola
                    # loss_vali = torch.mean(torch.mean(chi2_v**(1/2)))   # sqrt(chi2)

                    losses.append(np.float(loss_vali.cpu().detach().numpy()))

                losses_valid.append(np.mean(losses))

                scheduler.step(losses_valid[e])
                optim.zero_grad()

            self.update_progress(losses_train[-1],losses_valid[-1],train_start_time, e, n_epochs, optim)
        
        if ( save_losses ):
            np.savetxt("losses.txt", np.array([losses_train,losses_valid],dtype=np.float64))

        self.trained = True
        print('\nDone!')

    def predict(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            y_pred = self.model((torch.Tensor(X) - self.samples_mean) / self.samples_std)

        y_pred = (y_pred * self.dv_evals) @ torch.linalg.inv(self.dv_evecs) + self.dv_fid
        return y_pred.cpu().detach().numpy()

    def save(self, filename):
        #root = './external_modules/data/lsst_y1_cosmic_shear_emulator/'
        torch.save(self.model.state_dict(), filename)
        with h5.File(filename + '.h5', 'w') as f:
            f['sample_mean']   = self.samples_mean
            f['sample_std']    = self.samples_std
            f['dv_fid']        = self.dv_fid
            f['dv_evals']      = self.dv_evals
            f['dv_evecs']      = self.dv_evecs
        
    def load(self, filename, device=torch.device('cpu'),state_dict=True):
        #root = './external_modules/data/lsst_y1_cosmic_shear_emulator/'
        self.trained = True
        if device!=torch.device('cpu'):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if state_dict==False:
            self.model = torch.load(filename,map_location=device)
        else:
            state_dict = torch.load(filename, map_location=device)
            self.model.load_state_dict(state_dict)

        self.model.eval()

        with h5.File(filename + '.h5', 'r') as f:
            self.samples_mean  = torch.Tensor(f['sample_mean'][:])
            self.samples_std   = torch.Tensor(f['sample_std'][:])
            self.dv_fid        = torch.Tensor(f['dv_fid'][:])
            self.dv_evals      = torch.Tensor(f['dv_evals'][:])
            self.dv_evecs      = torch.Tensor(f['dv_evecs'][:])
        print('Loaded emulator')

