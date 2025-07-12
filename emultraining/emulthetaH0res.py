import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = "something"#Saving location for the model

#training setting
n_epoch = 700
input_size=len(train_samples[0])

#model dimension and layer
intdim = 50
Nlayer = 10


#load in data
train_samples=np.load('something',allow_pickle=True)
validation_samples=np.load('something',allow_pickle=True)
train_data_vectors=np.load('something',allow_pickle=True)
validation_data_vectors=np.load('something',allow_pickle=True)

input_size=len(train_samples[0])
out_size=1

extrainfo=np.load("something",allow_pickle=True)

class Supact(nn.Module):
    # New activation function, returns:
    # f(x)=(gamma+(1+exp(-beta*x))^(-1)*(1-gamma))*x
    # gamma and beta are trainable parameters.
    # I chose the initial value for gamma to be all 1, and beta to be all 0
    def __init__(self, in_size):
        super(Supact, self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(in_size))
        self.beta = nn.Parameter(torch.zeros(in_size))
        self.m = nn.Sigmoid()
    def forward(self, x):
        inv = self.m(torch.mul(self.beta,x))
        fac = 1-self.gamma
        mult = self.gamma + torch.mul(inv,fac)
        return torch.mul(mult,x)

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        # This function is designed for the Neuro-network to learn how to normalize the data between
        # layers. we will initiate gains and bias both at 1 
        self.gain = nn.Parameter(torch.ones(1))

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        return x * self.gain + self.bias


class ResMLP2(nn.Module):

    def __init__(self, input_dim, output_dim, int_dim, N_layer):

        super(ResMLP2, self).__init__()

        modules=[]

        # Def: we will set the internal dimension as multiple of 128 (reason: just simplicity)
        int_dim = int_dim

        # Def: we will only change the dimension of the datavector using linear transformations  
        modules.append(nn.Linear(input_dim, int_dim))
        
        # Def: by design, a pure block has the input and output dimension to be the same
        for n in range(N_layer):
            # Def: This is what we defined as a pure MLP block
            # Why the Affine function?
            #   R: this is for the Neuro-network to learn how to normalize the data between layer
            modules.append(ResBlock(int_dim, int_dim))
            modules.append(Supact(int_dim))
        
        # Def: the transformation from the internal dimension to the output dimension of the
        #      data vector we intend to emulate
        
        modules.append(nn.Linear(int_dim, output_dim))
        modules.append(Affine())
        # NN.SEQUENTIAL is a PYTHORCH function DEFINED AT: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # This function stacks up layers in the modules-list in sequence to create the whole model
        self.simpmlp =nn.Sequential(*modules)#

    def forward(self, x):
        #x is a cosmological parameter set you feed in the model
        out = self.simpmlp(x)

        return out

train_samples=torch.Tensor(train_samples)#.to(device)
train_data_vectors=torch.Tensor(train_data_vectors)#.to(device)
validation_samples=torch.Tensor(validation_samples)#.to(device)
validation_data_vectors=torch.Tensor(validation_data_vectors)#.to(device)

X_mean=torch.Tensor(extrainfo.item()['X_mean'])#.to(device)
X_std=torch.Tensor(extrainfo.item()['X_std'])#.to(device)
Y_mean=torch.Tensor(extrainfo.item()['Y_mean']).to(device)
Y_std=torch.Tensor(extrainfo.item()['Y_std']).to(device)

#normalizing samples and data vectors to mean 0, std 1

X_train=(train_samples-X_mean)/X_std

X_validation=(validation_samples-X_mean)/X_std

X_train=X_train.to(torch.float32)
X_validation=X_validation.to(torch.float32)
train_data_vectors=train_data_vectors.to(torch.float32)
validation_data_vectors=validation_data_vectors.to(torch.float32)


trainset    = TensorDataset(X_train, train_data_vectors)
validset    = TensorDataset(X_validation,validation_data_vectors)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

#Set up the model and optimizer

#training

model = RespMLP2(input_dim=input_size,output_dim=out_size,int_dim=intdim,N_layer=Nlayer)
    
model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)


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
        Y_pred = model(X).to(device)
        Y_pred_back = Y_pred*Y_std+Y_mean
        diff = (Y_batch[:,None] - Y_pred_back)
        
        loss1 = torch.diag(diff @ torch.t(diff))# implement with torch.einsum
        loss1 = torch.sqrt(1+2*loss1)
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
            Y_v_pred_back = Y_v_pred*Y_std+Y_mean
            v_diff = (Y_v_batch[:,None] - Y_v_pred_back)
            
            loss1 = torch.diag(v_diff @ torch.t(v_diff))# implement with torch.einsum
            loss1 = torch.sqrt(1+2*loss1)
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
                        
                    ))



torch.save(model.state_dict(), PATH+'.pt')
