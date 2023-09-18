# README DOCUMENTATION
Code inspiration:
- https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L17/1_VAE_mnist_sigmoid_mse.ipynb
- https://github.com/Michedev/VAE_anomaly_detection

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


----------------------------------
(TODO) Anomaly detection
    1. detect_alpha  -- based on training data (after training) and NORMAL LOSS function
    2. detect_anomalies -- based on test data (after training) and NORMAL LOSS function

    --> muss nicht in model.py, kann aber
    --> aktuell doppelt gemoppelt, da in model.py und anomaly_detection.py 

BUUUUG: Alle Instances eines Batches haben gleichen Loss, rip! 
        Grund: get_loss() returnt avg. loss (ein float, kein tensor)

        Lösung: in model.py 

        - erstelle:  def get_loss_of_instance(): Tensor
        - erstelle:  def get_loss_of_batch():float die get_loss_of_instance() aufruft und avg() oder sum()
                   -> äquivalent zum aktuellen get_loss(), die dadurch ersetzt wird
        
        - nutze get_loss_of_batch() in train()
        - nutze get_loss_of_instances in anomaly_detection.py

        + bitte in DOC String schreiben, dass eine durchschnitts-loss und eine isntance-loss berechnet

----------------------------------


(~) Evaluation
    (x) log all dictonaries (loss, etc.)
    (x) plot metrics
    (o) AUC ROC
    (o) AUC PR

(o) Reconstruction
    (o) plot latent space (e. g., images)
    (o) implement reconstructions with generate()
    (o) plot top 5 anomalies (e. g., images)
    (o) plot top 5 normals (e. g., images)





