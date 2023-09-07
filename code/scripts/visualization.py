import matplotlib.pyplot as plt
import torch

from helper_classes import pVAELogTrain

PATH:str = '../plots/'

def plot_train_preds(log_train_pred:pVAELogTrain):
    file_name:str = f'{PATH}training_progress.png'

    _, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(log_train_pred.loss)
    ax[0].set_title('Loss')
    ax[1].plot(log_train_pred.kl)
    ax[1].set_title('KL')
    ax[2].plot(log_train_pred.recon_loss)
    ax[2].set_title('Recon Loss')
    

    print(f'Plotting training progress in {file_name}\n')
    plt.savefig(file_name)
    return

def plot_mnist_orig_and_recon(
    batch_size:int,
    x_orig:torch.Tensor,
    x_recon:torch.Tensor,
    y:torch.Tensor,
)-> None: 
    file_name:str = f'{PATH}mnist_original_&_reconstruction.png'

    # show original and reconstructed images next to each other
    _, axes = plt.subplots(batch_size, 2, figsize=(10, 20))
    for i in range(batch_size):
        axes[i,0].imshow(x_orig[i].squeeze().detach().cpu().numpy(), cmap='gray')
        axes[i,1].imshow(x_recon[i].squeeze().detach().cpu().numpy(), cmap='gray')
        axes[i,0].set_title(f'Original {y[i]}')
        axes[i,1].set_title(f'Reconstructed {y[i]}')
        axes[i,0].axis('off')
        axes[i,1].axis('off')

    print(f'Plotting MNIST (original and reconstruction) in {file_name}\n')
    plt.savefig(file_name)
    pass