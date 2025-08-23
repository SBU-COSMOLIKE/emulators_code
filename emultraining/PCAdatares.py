import numpy as np
from sklearn.decomposition import IncrementalPCA
#This script works for models that are trained in ResMLP
#i.e H and PhiPhi
train_samples=np.load('training input',allow_pickle=True)# This is actually a latin hypercube sampling of 1mil points

train_data_vectors=np.log(np.load('training output',allow_pickle=True))

X_mean=train_samples.mean(axis=0, keepdims=True)
X_std=train_samples.std(axis=0, keepdims=True)
Y_mean=train_data_vectors.mean(axis=0, keepdims=True)
Y_std=train_data_vectors.std(axis=0, keepdims=True)


X=(train_data_vectors-Y_mean)/Y_std

n_pca=96
batchsize=43
PCA = IncrementalPCA(n_components=n_pca,batch_size=batchsize)

for batch in np.array_split(X, batchsize):
    PCA.partial_fit(batch)

train_pca=PCA.transform(X)

Y_mean_2=train_pca.mean(axis=0, keepdims=True)
Y_std_2=train_pca.std(axis=0, keepdims=True)

np.save('PCA Dir',PCA.components_)
extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean,'Y_std':Y_std,'Y_mean_2':Y_mean_2,'Y_std_2':Y_std_2}
np.save('Normalization Dir',extrainfo)