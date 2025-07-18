import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cobaya.theories.emulcmb.emulator import ResBlock, ResMLP, TRF, CNNMLP

camb_ell_min          = 2#30
camb_ell_max          = 3000
camb_ell_range        = camb_ell_max  - camb_ell_min 
out_size=camb_ell_range
batch_size=512
n_epoch=700#for trial test purpose
intdim=4 #internal dimension of the CNN

PATH = "Saving Directory, no .pt required"

As_pos = #index of log_A_s in the array of param
tau_pos = #index of tau in the array of param

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extrainfo = np.load("Normalization Factors",allow_pickle=True)
covinv = np.load('Covariance matrix I sent in mattermost',allow_pickle=True)[:camb_ell_range]*4/np.exp(4*0.06)
covinv = np.diag(covinv)
train_samples = np.load('Train set input',allow_pickle=True)

in_size=len(train_samples[0])
validation_samples = np.load('vali set input',allow_pickle=True)

train_data_vectors = np.load('train set output',allow_pickle=True,mmap_mode='r+')
# mmap_mode=r+ for huge files to be read from memory
# not in RAM
validation_data_vectors = np.load('vali set output ',allow_pickle=True)
X_mean = torch.Tensor(extrainfo.item()['X_mean'])#.to(device)
X_std  = torch.Tensor(extrainfo.item()['X_std'])#.to(device)
Y_mean = torch.Tensor(extrainfo.item()['Y_mean']).to(device)
Y_std  = torch.Tensor(extrainfo.item()['Y_std']).to(device)


covinv = torch.Tensor(covinv).to(device) #This is inverse of the Covariance Matrix

train_samples=torch.Tensor(train_samples)#.to(device)
train_data_vectors=torch.Tensor(train_data_vectors)#.to(device)
validation_samples=torch.Tensor(validation_samples)#.to(device)
validation_data_vectors=torch.Tensor(validation_data_vectors)#.to(device)


X_train=(train_samples-X_mean)/X_std
X_validation=(validation_samples-X_mean)/X_std
X_train=X_train.to(torch.float32)
X_validation=X_validation.to(torch.float32)

X_mean=X_mean.to(device)
X_std=X_std.to(device)

trainset    = TensorDataset(X_train, train_data_vectors)
validset    = TensorDataset(X_validation,validation_data_vectors)
#testingset    = TensorDataset(X_testing, testing_data_vectors)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
#testloader = DataLoader(testingset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

model = CNNMLP(input_dim=in_size, output_dim=out_size, int_dim=intdim)

model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0)

losses_train = []
losses_vali = []
losses_train_med = []
losses_vali_med = []

reduce_lr = True#reducing learning rate on plateau
if reduce_lr==True:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=15)

for n in range(n_epoch):
    losses=[]
    for i, data in enumerate(trainloader):
        model.train()

        X = data[0].to(device)# send to device one by one
            
        Y_batch = data[1].to(device)# send to device one by one
        Y_pred  = model(X).to(device)
            
        As = torch.exp(X[:,As_pos]*X_std[0,As_pos]+X_mean[0,As_pos])
        exptau = torch.exp(2*X[:,tau_pos]*X_std[0,tau_pos]+2*X_mean[0,tau_pos])
        Y_pred = Y_pred*Y_std+Y_mean
        Y_batch = Y_batch/As[:,None]*exptau[:,None]
        diff = Y_pred - Y_batch
        
        loss1 = torch.diag(diff @ covinv @ torch.t(diff))# implement with torch.einsum
        loss1 = torch.sqrt(loss1)
        loss = torch.mean(loss1)
        losses.append(loss.cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses_train.append(np.mean(losses))# We take means since a loss function should return a single real number
    losses_train_med.append(np.median(losses))

    with torch.no_grad():
        model.eval()
        
        losses = []
        for i, data in enumerate(validloader):
            X_v       = data[0].to(device)
                
            Y_v_batch = data[1].to(device)
            Y_v_pred = model(X_v).to(device)
            As=torch.exp(X_v[:,As_pos]*X_std[0,As_pos]+X_mean[0,As_pos])
            exptau=torch.exp(2*X_v[:,tau_pos]*X_std[0,tau_pos]+2*X_mean[0,tau_pos])
            Y_v_pred_back = Y_v_pred*Y_std+Y_mean
            Y_v_batch = Y_v_batch/As[:,None]*exptau[:,None]
            v_diff = (Y_v_batch - Y_v_pred_back)
            loss1 = torch.diag(v_diff @ covinv @ torch.t(v_diff))# implement with torch.einsum
            loss1 = torch.sqrt(loss1)
            loss_vali = torch.mean(loss1)
            losses.append(loss_vali.cpu().detach().numpy())

        losses_vali.append(np.mean(losses))
        losses_vali_med.append(np.median(losses))

        if reduce_lr == True:
            print('Reduce LR on plateu: ',reduce_lr)
            scheduler.step(losses_vali[n])
        

    print('epoch {}, loss={}, validation loss={}, lr={}, wd={})'.format(
                        n,
                        losses_train[-1],
                        losses_vali[-1],
                        optimizer.param_groups[0]['lr'],
                        optimizer.param_groups[0]['weight_decay']  
                    ))

torch.save(model.state_dict(), PATH+'.pt')

