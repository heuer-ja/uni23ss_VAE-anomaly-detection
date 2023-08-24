import matplotlib.pyplot as plt

from helper_classes import LogTrainPreds

def plot_train_preds(log_train_pred:LogTrainPreds):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(log_train_pred.loss)
    ax[0].set_title('Loss')
    ax[1].plot(log_train_pred.kl)
    ax[1].set_title('KL')
    ax[2].plot(log_train_pred.recon_loss)
    ax[2].set_title('Recon Loss')
    
    plt.savefig('training_progress.png')
    return