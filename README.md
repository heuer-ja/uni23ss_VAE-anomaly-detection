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

(~) Anomaly detection
    (x) implement anomaly detection
    (x) choose alpha - log all recon. probs. and choose max            
    (x) combine anomalies with label
        (x) BUG: MNIST nutzt bisher die Pixel (0-255) als Anomaly anstatt die Klassen (0-9)
        (x) BUG: KDD nutzt int für labels, da one-hot encoding, nicht str so wie ich es nutze
!!! (o) BUG: all instances are considered as anomalies 

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

---------------------------------------
(o) BUGFIX: 
    (o) TRAINING:
        (o) Ergebnisse Schrott! Erstmal das fixen      
        (o) nan loss when lr too high
            - batchnorm hilft

    (o) ANOM. DETECT:   all instances are considered as anomalies 
            - vermutlich weil Netzwerk nicht lernt und die max_alpha genommen wird während training, die dann wsh. random ist. Da Training mehr Instanzen hat, kann Testset solch eine random recon_prob nicht erreichen
