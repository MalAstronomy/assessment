import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

class Net(nn.Module):
    
    def __init__(self, n_channels_in, n_channels_out):
        super().__init__()
        
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_channels_in, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        
        # second hidden layer
        self.hidden2 = nn.Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        
        # third hidden layer and output
        self.hidden3 = nn.Linear(8, n_channels_out)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Sigmoid()
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X
    
#     def __init__(self, n_channels_in, n_channels_out):
#         super().__init__()
        
#         self.layer_in = self.layer(n_channels_in, 828)
#         self.layer_h1 = self.layer(828, 828)
#         self.layer_h2 = self.layer(828, 828)
#         self.layer_h3 = self.layer(828, 828)
#         self.layer_out = self.layer(828, n_channels_out, DO_prob=0) #no Dropout applied

        
#     def layer(self, nb_neurons_in, nb_neurons_out, DO_prob=0.5):
#         layer = nn.Sequential(
#             nn.Linear(nb_neurons_in, nb_neurons_out, bias=True),
#             nn.ReLU(),
#             #nn.SELU(),
#             nn.Dropout(p=DO_prob))
#         return layer
        

#     def forward(self, xb):
#         x = xb.view(xb.size(0), -1) #Flatten data

#         #print(x.size())
#         x = self.layer_in(x)
#         #print(x.size())
#         x = self.layer_h1(x)
#         x = self.layer_h2(x)
#         #x = self.layer_h3(x)
#         r = self.layer_out(x)

#         return r