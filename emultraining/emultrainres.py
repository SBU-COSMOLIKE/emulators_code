import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset
from cobaya.theories.emulcmb.emulator import ResBlock, ResMLP, TRF, CNNMLP
parser = argparse.ArgumentParser(prog='cos_uniform')


parser.add_argument("--batch",
                    dest="batch_size",
                    help="Number of samples per batch",
                    type=int,
                    nargs='?',
                    const=1,
                    default=512)
parser.add_argument("--epoch",
                    dest="n_epoch",
                    help="Number of epochs",
                    type=int,
                    nargs='?',
                    const=1,
                    default=700)

parser.add_argument("--intdim",
                    dest="intdim",
                    help="Number of internal dimension for the RsMLP, will be multiplied by 128",
                    type=int,
                    nargs='?',
                    const=1,
                    default=4)

parser.add_argument("--Nlayer",
                    dest="Nlayer",
                    help="Number of Layers of ResBlocks",
                    type=int,
                    nargs='?',
                    const=1,
                    default=5120)

parser.add_argument("--Path",
                    dest="PATH",
                    help="Saving Directory, no .pt required",
                    type=str,
                    nargs='?',
                    const=1,
                    default='model')

parser.add_argument("--PCA",
                    dest="PCA_file",
                    help="directory for PCA file",
                    type=str,
                    nargs='?',
                    const=1,
                    default='pca.npy')


parser.add_argument("--extrainfo",
                    dest="extrainfo_file",
                    help="directory for normalization factor file",
                    type=str,
                    nargs='?',
                    const=1,
                    default='extra.npy')



parser.add_argument("--traininput",
                    dest="train_param_file",
                    help="directory for training input file",
                    type=str,
                    nargs='?',
                    const=1,
                    default='train_input.npy')
parser.add_argument("--trainoutput",
                    dest="train_dv_file",
                    help="directory for training output file",
                    type=str,
                    nargs='?',
                    const=1,
                    default='train_output.npy')
parser.add_argument("--valiinput",
                    dest="vali_param_file",
                    help="directory for validation input file",
                    type=str,
                    nargs='?',
                    const=1,
                    default='vali_input.npy')
parser.add_argument("--valioutput",
                    dest="vali_dv_file",
                    help="directory for validation output file",
                    type=str,
                    nargs='?',
                    const=1,
                    default='vali_output.npy')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args, unknown = parser.parse_known_args()

PCA_file = args.PCA_file
extrainfo_file = args.extrainfo_file
train_param_file = args.train_param_file
vali_param_file = args.vali_param_file
train_dv_file = args.train_dv_file
vali_dv_file = args.vali_dv_file
PATH = args.PATH
batch_size = args.batch_size
n_epoch = args.n_epoch
Nlayer = args.Nlayer
intdim = args.intdim

#This Script can work for both Hubble emulator and Phiphi emulator. 
if __name__ == '__main__':
    

    transform_matrix=np.load(PCA_file,allow_pickle=True)#PCA transformation matrix
    transform_matrix = torch.Tensor(transform_matrix).to(device)
    extrainfo=np.load(extrainfo_file,allow_pickle=True)
    #load in data
    train_samples=np.load(train_param_file,allow_pickle=True)#.astype('float32')# This is actually a latin hypercube sampling of 1mil points

    input_size=len(train_samples[0])

    validation_samples=np.load(vali_param_file,allow_pickle=True)

    train_data_vectors=np.log(np.load(train_dv_file,allow_pickle=True))

    validation_data_vectors=np.log(np.load(vali_dv_file,allow_pickle=True))


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
            Y_batch = Y_batch
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
