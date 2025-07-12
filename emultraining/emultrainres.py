import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cobaya.theories.emulcmb.emulator import ResBlock, ResMLP, TRF, CNNMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#This Script can work for both Hubble emulator and Phiphi emulator. 
PATH = "Saving Directory, no .pt required"#saving location of the model
n_epoch = 700
intdim = 6 #internal dimension of ResMLP *128
Nlayer = 6 # number of layer of ResMLP

transform_matrix=np.load('PCA transformation matrix',allow_pickle=True)#PCA transformation matrix
transform_matrix = torch.Tensor(transform_matrix).to(device)
extrainfo=np.load("Normalization factors",allow_pickle=True)
#load in data
train_samples=np.load('training set input',allow_pickle=True)#.astype('float32')# This is actually a latin hypercube sampling of 1mil points

input_size=len(train_samples[0])

validation_samples=np.load('vali set input',allow_pickle=True)

train_data_vectors=np.load('training set output',allow_pickle=True,mmap_mode='r+')

validation_data_vectors=np.log(np.load('vali set output',allow_pickle=True))


out_size=len(transform_matrix)

train_samples=torch.Tensor(train_samples)#.to(device)
train_data_vectors=torch.Tensor(train_data_vectors)#.to(device)
validation_samples=torch.Tensor(validation_samples)#.to(device)
validation_data_vectors=torch.Tensor(validation_data_vectors)#.to(device)


X_mean=torch.Tensor(extrainfo.item()['X_mean'])#.to(device)
X_std=torch.Tensor(extrainfo.item()['X_std'])#.to(device)
Y_mean=torch.Tensor(extrainfo.item()['Y_mean']).to(device)
Y_std=torch.Tensor(extrainfo.item()['Y_std']).to(device)
Y_mean_2=torch.Tensor(extrainfo.item()['Y_mean_2']).to(device)
Y_std_2=torch.Tensor(extrainfo.item()['Y_std_2']).to(device)

#normalizing samples and data vectors to mean 0, std 1

X_train=(train_samples-X_mean)/X_std

#print(train_data_vectors.dtype)
#print(X_train.dtype)
X_validation=(validation_samples-X_mean)/X_std

X_train=X_train.to(torch.float32)
X_validation=X_validation.to(torch.float32)


batch_size=512
trainset    = TensorDataset(X_train, train_data_vectors)
validset    = TensorDataset(X_validation,validation_data_vectors)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

#Set up the model and optimizer

#training

model = ResMLP(input_dim=input_size,output_dim=out_size,int_dim=intdim,N_layer=Nlayer)
    

model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters())


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
        X       = data[0].to(device)

        Y_batch = data[1].to(device)
        Y_batch = torch.log(Y_batch)
        Y_pred = model(X).to(device)
            
        Y_pred_back = torch.matmul((Y_pred*Y_std_2+Y_mean_2),transform_matrix)*Y_std+Y_mean
        diff = (Y_batch - Y_pred_back)
        

        
        loss1 = torch.diag(diff @ torch.t(diff))# implement with torch.einsum
        loss1 = torch.sqrt(loss1)
        loss=torch.mean(loss1)
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
            Y_v_pred_back = torch.matmul((Y_v_pred*Y_std_2+Y_mean_2),transform_matrix)*Y_std+Y_mean
            v_diff = (Y_v_batch - Y_v_pred_back)
            
            loss1 = torch.diag(v_diff @ torch.t(v_diff))# implement with torch.einsum
            loss1 = torch.sqrt(loss1)
            loss_vali=torch.mean(loss1)
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
                        
                    ))#, total runtime: {} ({} average))

torch.save(model.state_dict(), PATH+'.pt')
