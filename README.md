# README DOCUMENTATION
# Execution
CUDA_VISIBLE_DEVICES=0,1 nohup python main.py > log.txt


# TODOs
(x) MNIST (image)
    (x) load data
    (x) model CNN
    (x) train
    (x) main
    (x) split into normal and anomaly
        (x) download train and test
        (x) from train: extract anomaly class
        (x) add anomaly class to test
        (x) return X (only normals), and y (normals & anomalies)
 
(x) KDD1999 (tabular)
    (x) load data
    (x) model Fully connected
    (x) train
    (x) main
        (x) split into data into train and test
        (x) from train: extract anomaly class
        (x) add anomaly class to test
        (x) return X (only normals), and y (normals & anomalies)

(x) Implemenet normal VAE
    (x) main.py anpassen an normal VAE
    (x) visualazation.py anpassen an normal VAE
    (FAIL) dVAE need to use L
    (x) delete self.L 
    (x) update loss in dVAE: KL divergence + MSE or CrossEntropy



(x) Evaluation
    (x) log all dictonaries (loss, etc.)
    (x) plot metrics
    (x) AUC ROC
        (x) evaluation.py
            (x) recon_loss von test data erhalten
            (x) verbinde recon_loss mit labels
            (x) normalize recon_loss [0,1] falls notwendig (ist nicht notwendig!)
            (x) roc_curve berechnen (fpr, tpr, thresholds)
            (x) auc berechnen
            (x) plot roc_curve 
            --> roc curve testet sogesehen k verschiedene alphas (normalisiert oder nicht) 

    (x) AUC PR
    (x) F1 Score

(x) Anomaly detection
    (x) detect_alpha -- based on training data (after training) and REC. LOSS function
    (x) detect_anomalies -- based on test data (after training) and REC. LOSS function
    (x) return DataFrame holding information about anomaly distribution

(x) create run.py and new_main.py
    (x) rename main.py -> run.py
    (x) new main: iterate run.py for different anomaly classes
    (x) create a DF holding all information (auc_roc, f1, auc_prc) about the different anomaly classes

(x) BUGFIXES: 
    (x) dVAE KDD1999 not working (stops at evaluation.py in for loop)
    (x) plot_mnist_orig_and_recon error MNIST.Six
    --> habs behoben in train, siehe GitLens. 
        Erklarung: in letztem loop sind nur Restinstanzen, dessen Anzahl nicht batch_size ist sondern weniger, im Falle von MNIST.Six = 2.  

        Loesung: Schon behoben, aber plot sind dann halt crap aus mit nur 2 Bildern

(x) Extra
    (x) move roc_plot to visualazation.py
    (x) add col to df showing how much % of anomalies are in test data
    (x) save output df as csv


(WIP) Jupyter Notebook
    (x) create jupyter notebook on server
    (x) imports on top
    (x) Introduction text on top
        (x) why pVAE is not working, but dVAE is working\
        (x) what is this code about
        (x) model architecture + flaws in paper not mentoining the architecture preciscly

    (x) Code
    (x) plot visuals somehow if neccessary
    (WIP) Results text on top HIER WEITERMACHEN!!
    (o) Anomaly Detection machen


(o) optional: Reconstruction
    (o) plot latent space (e. g., images)
    (x) implement reconstructions with generate()
    (o) plot top 5 anomalies (e. g., images)
    (o) plot top 5 normals (e. g., images)


# Code inspiration:
- https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L17/1_VAE_mnist_sigmoid_mse.ipynb
- https://github.com/Michedev/VAE_anomaly_detection

