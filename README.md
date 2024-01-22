# Retrieving data

Logs of random walks of Sigmabans can be found [here](https://drive.google.com/drive/folders/1U7Teez7FN7PS6OrcHcADJV89jdhkNELz?usp=drive_link). Place the raw and the processed logs (to gain computation time or to use train without having Sigmaban URDF) in the logs folder.

# Build datasets

Use **python log_processing -s "size"** to builds datasets with "size" history length.

# Train the data

Use **python learn.py -s "size"** to train your NN.

optional : 

* -w 1 to use with wandb
* -a to change the activation function
* -e to change the number of epochs

You can also use the shell scipt **./train.sh** to train with several options.