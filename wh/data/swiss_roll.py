####################################
### Make Swiss Roll Dataset ########
####################################

import numpy as np
from sklearn.datasets import make_swiss_roll
import torch
import torch.nn.functional as F
import torch.utils.data as utils



def make_swiss_roll_dataset(samples=1e4, batch_size=32):
    """
    Create swiss roll dataset and return with the loaders.
    """
    x_train, manifold_x_train = make_swiss_roll(n_samples=samples)
    x_train = x_train.astype(np.float32)
    y_train = (x_train[:, 0:1] >= 10).astype(np.float32)

    train_set_x_tensor = torch.from_numpy(x_train).double()
    train_set_y_tensor = F.one_hot(torch.from_numpy(y_train).flatten().type(torch.int64)).double()
    train_set = utils.TensorDataset(train_set_x_tensor, train_set_y_tensor)
    train_dataloader = utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    x_test, manifold_x_test = make_swiss_roll(n_samples=samples)
    x_test = x_test.astype(np.float32)
    y_test = (x_test[:, 0:1] >= 10).astype(np.float32)

    test_set_x_tensor = torch.from_numpy(x_test).double()
    test_set_y_tensor = F.one_hot(torch.from_numpy(y_test).flatten().type(torch.int64)).double()
    test_set = utils.TensorDataset(test_set_x_tensor, test_set_y_tensor)
    test_dataloader = utils.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_set, train_dataloader, test_set, test_dataloader