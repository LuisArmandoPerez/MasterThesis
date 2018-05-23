#!/usr/bin/anaconda3/bin/python3
import os
import sys
from imageio import imread
import numpy as np
import modules.vae_nn_imageshift_cluster

# Get the
sys.path.append(os.getcwd())

# RELEVANT PATH ADDRESSES FOR SAVING
cwd = os.getcwd()
log_dir_tensorboard = os.path.join(cwd, 'tensorboard')
weight_dir = os.path.join(cwd, 'model_weights')
weight_file = os.path.join(cwd, 'model_weights', 'model.cpkt')
image_file = os.path.join(cwd, 'resources', 'symbol4.png')

# LOAD IMAGE
image = imread(image_file)
image = image[:, :, 3] / np.max(image)  # Normalize image

# VAE
epochs = 10000
axis = 1
batch_size = 32
learning_rate = 0.0001
vae_parameters = {'latent_dim': 2,
                  'mode': {'encoder': 'Normal'},
                  'learning_rate': learning_rate,
                  'shape': image.shape}
vae = modules.vae_nn_imageshift_cluster.vae_nn_images(**vae_parameters)
train_parameters = {'image': image,
                    'axis': axis,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'log_dir_tensorboard': log_dir_tensorboard,
                    'weights_folder': weight_file,
                    'shuffle': True}
vae.train_shift_images(**train_parameters)
print("DONE TRAINING")
