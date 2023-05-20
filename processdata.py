import torch
from scipy.io import loadmat
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

class TimeSeriesDataset(torch.utils.data.Dataset):
    '''Takes input sequence of sensor measurements with shape (batch size, lags, num_sensors)
    and corresponding measurments of high-dimensional state, return Torch dataset'''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len

def load_data(name):
    '''Takes string denoting data name and returns the corresponding (N x m) array 
    (N samples of m dimensional state)'''
    if name == 'SST':
        load_X = loadmat('Data/SST_data.mat')['Z'].T
        mean_X = np.mean(load_X, axis=0)
        sst_locs = np.where(mean_X != 0)[0]
        return load_X[:, sst_locs]
    
    if name == 'FULL_SST':
        load_X = loadmat('Data/SST_data.mat')['Z'].T
        return load_X
    
    if name == 'AO3':
        load_X = np.load('Data/short_svd_O3.npy')
        return load_X

    if name == 'ISO':
        load_X = np.load('Data/numpy_isotropic.npy').reshape(-1, 350*350)
        return load_X

def plot_sst(x1, x2):
    '''Take in two full SST data matrix and plot world map projection'''
    for x in [x1, x2]:
        x.resize(len(x),180,360)
    
    index = np.random.randint(len(x1))
    plotdata = [x1, x2]
    labels = ['truth', 'recon']
    fig, ax = plt.subplots(1, 2, constrained_layout=True, sharey=True, figsize=(10,10))
    for axis,p,label in zip(ax, plotdata, labels):
        axis.imshow(p[index])
        axis.set_aspect('equal')
        axis.text(0.1,0.1,label,color='w',transform=axis.transAxes)

def qr_place(data_matrix, num_sensors):
    '''Takes a (m x N) data matrix consisting of N samples of an m dimensional state and
    number of sensors, returns QR placed sensors and U_r for the SVD X = U S V^T'''
    u, s, v = np.linalg.svd(data_matrix, full_matrices=False)
    rankapprox = u[:, :num_sensors]
    q, r, pivot = scipy.linalg.qr(rankapprox.T, pivoting=True)
    sensor_locs = pivot[:num_sensors]
    return sensor_locs, rankapprox

