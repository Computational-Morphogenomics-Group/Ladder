"""The visuals module houses the visualizers for workflows."""

import matplotlib.pyplot as plt


# Visualizer to plot model losses
def _plot_loss(train_losses, test_losses, save_loss_path: str = None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].plot(train_losses)
    ax[1].plot(test_losses)
    ax[0].set_title("Loss - Training")
    ax[1].set_title("Loss - Test")

    if save_loss_path is not None:
        plt.savefig(save_loss_path, dpi=300, bbox_inches="tight")

    fig.supxlabel("Epochs")
    fig.supylabel("Loss")

    plt.show()
