#!/usr/bin/env python

# Example of network training using Pytorch (1.1) and Cuda (9).

# NB: This code use real time monitoring based on Visdom
# an open source webserver allowing real time monitoring
#
# https://github.com/facebookresearch/visdom
#
# Start the webserver using:
# python -m visdom.server
#
# Access it on: (by default)
# http://localhost:8097

# Global import

import sys, os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from collections import OrderedDict

import click

# Local import
sys.path.insert(0, '/home/mvasist/scripts/assessment/training/methods/')
# Import the neural network architectures:
from MLP import Net as MLP
#from SCN import Model as SCN
#from CNN_3CV import Net as CNN

sys.path.insert(0, '/home/mvasist/scripts/assessment/training/src/')
from dataset import spectrum_dataset, splitDataLoader, ToTensor, Normalize
from criterion import *
import utils
import json
import logging


def train(model, dataset, optimizer, criterion, model_name, split=[0.9, 0.1], batch_size=32, 
          n_epochs=1, model_path='/home/mvasist/scripts/assessment/models/', random_seed=None):
    print('im in')
    # Create directory if doesn't exist
    model_dir = model_path+model_name 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Logging: we save output mmessages in a log file
    log_path = os.path.join(model_dir, 'logs.log')
    utils.set_logger(log_path)
    
    print('datasets')
    # Dataset
    dataloaders = {}
    dataloaders['train'], dataloaders['val'] = splitDataLoader(dataset, split=split, 
                                                               batch_size=batch_size, random_seed=random_seed)

    # ---
    # If the validation loss reaches a plateau, we decrease the learning rate:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    #scheduler = CosineWithRestarts(optimizer, T_max=40, eta_min=1e-7, last_epoch=-1)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7, last_epoch=-1)
    
    # Metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    
    metrics = {
        'model': model_dir,
        'optimizer': optimizer.__class__.__name__,
        'criterion': criterion.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'dataset_size': int(len(dataset)),
        'train_size': int(split[0]*len(dataset)),
        'test_size': int(split[1]*len(dataset)),
        'n_epoch': n_epochs,
        'batch_size': batch_size,
        'learning_rate': [],
        'train_loss': [],
        'val_loss': []
    }
    
    # Device: We use cuda on GPUs only
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    # Training
    since = time.time()
    dataset_size = {
        'train':int(split[0]*len(dataset)),
        'val':int(split[1]*len(dataset))
    }
      
    best_loss = 0.0
    for epoch in range(n_epochs):
        print(epoch, ' epoch')
        logging.info('-'*30)
        epoch_time = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            print(phase)
            for _, sample in enumerate(dataloaders[phase]):
                # GPU support 
                inputs = sample['input'].to(device)
                target = sample['target'].to(device)
                print('1')
                ##############################################################
                
                # Zero the parameter gradients
                # The backward() function accumulates gradients -> zero_grad() not to mix up gradients between minibatches
                optimizer.zero_grad()
                
                # forward: track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # 1. Make prediction:
                    ratio_estimation = model(inputs)
                    #print(ratio_estimation)
                    #print(abs(ratio_estimation - target))
                    # 2. Compute the loss for the current batch:
                    loss = criterion(torch.squeeze(ratio_estimation), torch.squeeze(target))
                    print('2')
                    # Perform backward propagation to update the weights:
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   
                    
                    running_loss += 1 * loss.item() * inputs.size(0)
                    
            logging.info('[%i/%i] %s loss: %f' % (epoch+1, n_epochs, phase, running_loss / dataset_size[phase]))
            print('3')
            # Update metrics
            metrics[phase+'_loss'].append(running_loss / dataset_size[phase])
            #metrics['zernike_'+phase+'_loss'].append(zernike_loss / dataset_size[phase])
            if phase=='train':
                metrics['learning_rate'].append(get_lr(optimizer))
                
            # Adaptive learning rate
            if phase == 'val':
                #scheduler.step()
                # If scheduler is ReduceLROnPlateau we need to give current validation loss:
                scheduler.step(metrics[phase+'_loss'][epoch])
                # Save weigths
                if epoch == 0 or running_loss < best_loss:
                    best_loss = running_loss
                    model_path = os.path.join(model_dir, 'model.pth')
                    torch.save(model.state_dict(), model_path)
                # Save metrics
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4) 
                print('4')    
        print('phase out ', phase)                    
        logging.info('[%i/%i] Time: %f s' % (epoch + 1, n_epochs, time.time()-epoch_time))
        
    time_elapsed = time.time() - since    
    logging.info('[-----] All epochs completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            
        
def get_lr(optimizer):
    for p in optimizer.param_groups:
        lr = p['lr']
    return lr  

#### Input options ######
@click.command()

## Mandatory:
@click.option('-dla','--dl_arch', type=str, 
              help='Machine learning architecture selected', required=True)
@click.option('-dsiz','--dataset_size', type=int, 
              help='Number of spectra used for training and validation', required=True)
@click.option('-floc','--file_location', type=str, 
              help='Location of the training dataset file', required=True)
@click.option('-nparam','--number_of_parameters', type=int, 
              help='Number of parameters for retrieval', required=True)


## Optional:
@click.option('-opt','--optimizer_name', type=str, 
              help='Name of the optimizer', default='Adam')
@click.option('-bs','--batch_size', type=int, 
              help='Batch size', default=64)
@click.option('-lr','--learning_rate', type=str, 
              help='Learning rate', default='1e-3')
@click.option('-nep','--nb_epoch', type=int, 
              help='Number of epochs', default=200)
@click.option('-splt','--split_train', type=str, 
              help='Proportion of spectral parameters used for training (the rest is for validation)', default='0.9')
@click.option('-met','--metric', type=str, 
              help='Metric to use to compute the loss during training', default='mse')
              
def main(dataset_size, file_location, dl_arch, optimizer_name, batch_size, learning_rate, nb_epoch, split_train, metric, number_of_parameters):    
    
    print('in the main')
    # Name to give to the model file:
    model_file_name = 'model_'+str(dataset_size)+'spectrum_'+dl_arch+'_bs'+str(batch_size)+'_lr'+\
    str(learning_rate)+'_'+str(nb_epoch)+'ep_opt'+str(optimizer_name)+'_split'+split_train+'_'+metric+'_'
    
    # Path where the model will be located:
    res_path = '/home/mvasist/scripts/assessment/models/'

    transfo = transforms.Compose([Normalize(), ToTensor()])

    # Create the dataset object:xc
    dataset = spectrum_dataset(path_to_file = file_location, 
                              size = int(float(dataset_size)),
                              transform = transfo)

    # Load model architecture:
    if dl_arch == 'mlp':
        model = MLP(number_of_parameters+371, 1)   #for mse (n+371, 2) target - [0 1]
    elif dl_arch == 'cnn':
        model = CNN(1, 2)
    else:
        raise ValueError("the model name specified is not valid")

    # Select which criterion to use to compute the loss:
    if metric == 'mse':
        criterion = MSELoss()
#     elif metric == 'rmse':
#         criterion = RMSELoss()
#     elif metric == 'mae':
#         criterion = MAELoss()
    elif metric == 'bce': 
        criterion = BCELoss()
    elif metric == 'ce': 
        criterion = CrossEntropyLoss()
    else:
        raise ValueError("the criterion name specified is not valid")

    # Select which optimizer to use:
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=float(learning_rate), momentum=0.9)
    else:
        raise ValueError("the optimizer specified is not valid")

    # Move Network to GPU if available:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('###########', device, '###########')
    if device.type == 'cuda':
        model.cuda()
    else:
        model.cpu()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()

    # Launch training script. The network weights are automatically saved 
    # at the end of an epoch (if the test error is reduced). The metrics are also
    # saved at the end of each epoch in JSON format. All outputs are also stored in a 
    # log file.
    #
    # - model = network to train
    # - dataset = dataset object
    # - optimizer = gradient descent optimizer (Adam, SGD, RMSProp)
    # - criterion = loss function
    # - split[x, 1-x] = Division train/test. 'x' is the proportion of the test set.
    # - batch_size = batch size
    # - n_epochs = number of epochs
    # - model_dir = where to save the results
    # - visdom =  enable real time monitoring

    #Launch training:

    train(model, 
          dataset,
          optimizer, 
          criterion,
          model_name = model_file_name,
          split = [float(split_train)/100, 1-float(split_train)/100], # Split between training and validation dataset
          batch_size = batch_size, # number limite by GPU
          n_epochs = nb_epoch,
          model_path = res_path)

if __name__ == '__main__':
    main()

