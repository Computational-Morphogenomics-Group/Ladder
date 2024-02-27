########################################
# Code used to generate synthetic data
# Auth: @zgr2788 
# TODO: Add more


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


########################################



def generate_simulated_data(d, t, n, p_deg=0, c_s=1, a_s=1, idx_s=1, e_m=0, e_s=1, linear=True):
    """
    Parameters
    ----------
    d : int
        number of features for generated points
    t : int
        number of time points to generate
    n : int
        number of points to generate
    p_deg : int, optional
        degree of the polynomial to be used, has no effect if ``linear==True``
    c_s : float, optional
        factor used to scale coefficients
    a_s : float, optional
        factor used to scale idx_s, e_m and e_s together
    idx_s : float, optional
        factor used to scale time indices when generating, higher values indicate greater separation along timepoints
    e_m : float,optional
        error mean
    e_s : float, optional
        error std

        
    Returns
    -------
    data : array-like, shape (n, t, d)
        generated points across time
    poly_coeffs : array-like, shape (n, d-1)
        log dictionary return only if log==True in parameters

    Examples
    --------
    
    """
    n *= 2 
    data = np.zeros((n, t, d))

    if linear:
        poly_coeffs = np.random.normal(0,1,size=(d - 1)) * c_s

    else:
        poly_coeffs = np.random.normal(0,1,size=(d-1, p_deg)) * c_s


    
    for i in range(n):
        index = np.arange(t).reshape(-1, 1) * idx_s * a_s
        
        if linear:
            poly_terms = index.dot(poly_coeffs.reshape(1, -1))

            
        else:
            idx_flat = index.flatten()
            poly_terms = np.array([[np.polyval(poly_coeffs[j], idx_flat[i]) for j in range(d-1)]  for i in range(t)])



        noise = np.random.normal(e_m * a_s, e_s * np.sqrt(a_s), (t, d))
        
        data[i, :, :] = np.concatenate([index, poly_terms], axis=1) + noise
            
    return data, poly_coeffs






class SyntheticDataset(Dataset):
    """
    Dataset object to use with generated points.
    """
    def __init__(self, dat_lib,  num_classes=None, transform=None, target_transform=lambda x : x.double()):
        self.dat_lib = dat_lib
        
        if num_classes:
            assert num_classes >= self.dat_lib.shape[0]
            self.labels = torch.tensor(np.array([[0]*i + [1] + [0]*(num_classes-1-i) for i in range(num_classes)]))  # Generate one hot encoding for all classes

        else:
            self.labels = torch.tensor(np.array([[0]*i + [1] + [0]*(self.dat_lib.shape[0]-1-i) for i in range(self.dat_lib.shape[0])]))  # Generate one hot encoding for all classes

            
        self.matched_dat = [(elem, self.labels[i]) for i in range(self.dat_lib.shape[0]) for elem in self.dat_lib[i]] # Match encodings to classes and flatten dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.matched_dat)

    def __getitem__(self, idx):

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step            
            vector, label = torch.vstack([elem[0] for elem in self.matched_dat[start:stop:step]]), torch.vstack([elem[1] for elem in self.matched_dat[start:stop:step]])
        
        else:
            vector, label = self.matched_dat[idx]
        
        if self.transform:
            vector = self.transform(vector)
        
        if self.target_transform:
            label = self.target_transform(label)

        return vector, label




def make_lin_pol_dataset(*args, **kwargs):
    """
    Make dataset object and return with loaders.
    """
    n_samples = args[2]
    batch_size = kwargs.get("batch_size", 2**4)

    
    data, _ = generate_simulated_data(*args, **kwargs)

    data = np.einsum("ijk -> jik", data)
    
    
    dat_lib_train, dat_lib_test = torch.tensor(data[:,:n_samples,:]), torch.tensor(data[:,n_samples:n_samples*2,:]) 

    # Prepare loaders
    train_dataset = SyntheticDataset(dat_lib_train)
    test_dataset = SyntheticDataset(dat_lib_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, train_dataloader, test_dataset, test_dataloader