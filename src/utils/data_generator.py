import torch
import numpy as np
from utils import *

def get_data(data_type="lilac"):
    
    path_sub =  f"../data/{data_type}/"
    print(f"Load {data_type} aggregated data from {path_sub}")
    current = np.load(path_sub+"current.npy", allow_pickle=True)
    voltage = np.load(path_sub+"voltage.npy",allow_pickle=True)
    label = np.load(path_sub+"labels.npy", allow_pickle=True)
    i_max = np.load(path_sub+"i_max.npy", allow_pickle=True)
    
    return current, voltage, label, i_max

class Dataset(torch.utils.data.Dataset):
    

    def __init__(self, feature, i_max, label):
       
        self.feature   = feature
        self.label    = label
        self.i_max    = i_max
        
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
       
        feature = self.feature[index]
        i_max   = torch.tensor(self.i_max[index]).float()
        label =  self.label[index]
       
        
        return feature,  i_max, label
        
        
def get_loaders(input_tra, input_val, i_max_tra, i_max_val, label_tra, label_val, batch_size):
   
    tra_data = Dataset(input_tra,i_max_tra, label_tra)
    val_data = Dataset(input_val, i_max_val, label_val)
    
    tra_loader=torch.utils.data.DataLoader(tra_data, batch_size, shuffle=True, num_workers=4)
    val_loader=torch.utils.data.DataLoader(val_data, batch_size, shuffle=False, num_workers=4)
    
    loaders = {'train':tra_loader, 'val':val_loader}
    
    return loaders  
