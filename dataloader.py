from torch.utils.data import Dataset
import numpy as np
import torch

def loadDataset(type="full", stride=None, len_samples = 2**15):
    
    if type == "full" and stride != None:
        data_train = torch.Tensor(np.load('./data/full_signal.npy'))
        dataset = DatasetCustomFull(data_train, stride, len_samples=len_samples)
        n_samples = dataset.n_samples
    elif type == "segmented":
        data_train = torch.Tensor(np.load('./data/data.npy'))
        dataset = DatasetCustom(data_train)
        n_samples = dataset.n_samples
    else:
        return None, None, None

    return dataset, n_samples, len_samples

# Define DataLoader
class DatasetCustom(Dataset):
    def __init__(self, data):
        # Data should be an array 
        self.data = data
        self.n_samples = self.data.shape[0]

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return np.expand_dims( self.data[index,:], axis=0)
        
class DatasetCustomFull(Dataset):
    def __init__(self, data, stride, len_samples=2**15):
        # Data should be an array 
        self.data = data
        self.len_samples = len_samples
        self.stride = stride
        self.n_samples = int(np.floor((self.data.shape[0] - len_samples) / stride))

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        return np.expand_dims( self.data[index*self.stride: index*self.stride + self.len_samples], axis=0)
        