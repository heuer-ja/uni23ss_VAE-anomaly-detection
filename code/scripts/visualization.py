import matplotlib.pyplot as plt

from helper_classes import LogTrainPreds

def plot_train_preds(log_train_pred:LogTrainPreds):
    file_name:str = 'training_progress.png'

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(log_train_pred.loss)
    ax[0].set_title('Loss')
    ax[1].plot(log_train_pred.kl)
    ax[1].set_title('KL')
    ax[2].plot(log_train_pred.recon_loss)
    ax[2].set_title('Recon Loss')
    

    print(f'Plotting training progress in {file_name}\n')
    plt.savefig(file_name)
    return