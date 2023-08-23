# README DOCUMENTATION
Code inspiration:
- https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L17/1_VAE_mnist_sigmoid_mse.ipynb
- https://github.com/Michedev/VAE_anomaly_detection

# Execution
CUDA_VISIBLE_DEVICES=0,1 nohup python main.py > log.txt

# Notes an mich
Hey, der Code bricht wegen Shape Problem. Tut mir sehr leid für dich.
Also, was du machen könntest:


    Probleme:
        - NaNs
        bei einer zu hohen learning rate, entstehen NaNs bei MNIST. Geht weg ab 5e-8, aber nach 18 epochen dann trotzdem
    
    Keine Probleme:
        - BATCH SIZE
        es wird ganzer Datensatz genutzt, also alle 60k Bilder. Im log steht zB Batch 0000/0469. Das ist weil man 60.000 / batchsize 

        bsp: Batch 0000/0469, da 60000/128 = 469\


# TODOs
        (x) MNIST (image)
            (x) load data
            (x) model CNN
            (x) train
            (x) main

        (x) KDD1999 (tabular)
            (x) load data
            (x) model CNN
            (x) train
            (x) main

        (o) Anomaly detection
            (o) implement anomaly detection

        (o) Evaluation
            (o) plot metrics
            (o) plot latent space (e. g., images)
            (o) plot reconstructions (e. g., images)
            (o) plot top 5 anomalies (e. g., images)
            (o) plot top 5 normals (e. g., images)
        
