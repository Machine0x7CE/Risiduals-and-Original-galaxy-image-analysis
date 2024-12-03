import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

batch_size = 64
nworkers = 8



def get_dataloader():
    # Load the dataset
    data = np.load('norm-res-95.npy')
    data = data.reshape(10046, 64, 64)
    data_tensor = torch.Tensor(data).unsqueeze(1)  # Add channel dimension
    dataset = TensorDataset(data_tensor)
    
    # Split the dataset into train and validation
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader



'''
def get_dataloader():
    # Load the dataset
    global train_dataloader, val_dataloader  # Use global variables
    data = np.load('norm-res-95.npy')
    data = data.reshape(10046, 64, 64)
    data_tensor = torch.Tensor(data).unsqueeze(1)  # Add channel dimension
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # DataLoader
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nworkers, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nworkers, persistent_workers=True)
    return dataloader, train_dataloader, val_dataloader
'''