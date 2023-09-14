import matplotlib.pyplot as plt
import torch

from helper_classes import VAELogTrain

PATH:str = '../plots/'


def plot_train_pred(log_train_pred:VAELogTrain, is_probabilistic:bool):
    if is_probabilistic:
        _plot_pVAE_train_pred(log_train_pred)
    else:
        _plot_VAE_train_pred(log_train_pred)
    return


def _plot_VAE_train_pred(log_train_pred:VAELogTrain):
    log_train_pred.loss = torch.stack(log_train_pred.loss).cpu().detach().numpy()

    file_name:str = f'{PATH}training_progress.png'

    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(range(len(log_train_pred.loss)), log_train_pred.loss, s=1)
    ax.set_title('Loss')
    
    print(f'Plotting training progress in {file_name}\n')
    plt.savefig(file_name)
    return


def _plot_pVAE_train_pred(log_train_pred:VAELogTrain):
    log_train_pred.loss = torch.stack(log_train_pred.loss).cpu().detach().numpy()
    log_train_pred.kl = torch.stack(log_train_pred.kl).cpu().detach().numpy()
    log_train_pred.recon_loss = torch.stack(log_train_pred.recon_loss).cpu().detach().numpy()

    file_name:str = f'{PATH}training_progress.png'

    _, ax = plt.subplots(1, 3, figsize=(15, 5))

    #turn into scatter plot 
    ax[0].scatter(range(len(log_train_pred.loss)), log_train_pred.loss, s=1)
    ax[0].set_title('Loss')
    ax[1].scatter(range(len(log_train_pred.kl)), log_train_pred.kl, s=1)
    ax[1].set_title('KL')
    ax[2].scatter(range(len(log_train_pred.recon_loss)), log_train_pred.recon_loss, s=1)
    ax[2].set_title('Reconstruction Loss')

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