import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from umap import UMAP


def plot_loss(train_losses, test_losses, save_loss_path : str = None):
    """
    Plots loss functions for the workflows.
    """
    fig,ax = plt.subplots(1,2, figsize = (10,5), sharex = True, sharey=True)
    ax[0].plot(train_losses) ; ax[1].plot(test_losses)
    ax[0].set_title("Loss - Training") ; ax[1].set_title("Loss - Test")

    if save_loss_path is not None:
        plt.savefig(save_loss_path, dpi=300, bbox_inches='tight')

    fig.supxlabel("Epochs")
    fig.supylabel("Loss")

    plt.show()