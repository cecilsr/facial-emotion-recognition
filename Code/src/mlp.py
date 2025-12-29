"""Multilayer perceptron for facial emotion recognition."""


import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


nodes_fc1 = 132
nodes_fc2 = 2048
out_classes = 7

class MLP(nn.Module):
    """Fully connected neural network for facial emotion recognition.

    The network flattens RGB images of shape (3, 100, 100) and outputs logits
    for 7 emotion classes.

    Parameters
    ----------
    dr_rate : float, optional
        Dropout rate applied after the first hidden layer.
    """
    def __init__(self, dr_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3*100*100, nodes_fc1)    # first hidden layer 
        self.fc2 = nn.Linear(nodes_fc1, nodes_fc2)        # second hidden layer 
        self.fc3 = nn.Linear(nodes_fc2, out_classes)  # output layer 
        self.dropout = nn.Dropout(dr_rate)

    def forward(self, x):
        x = x.view(x.size(0), -1)     # flatten images into vectors 
        
        x = F.relu(self.fc1(x))       # hidden layer 1 + ReLU activation
        x = self.dropout(x)

        x = F.relu(self.fc2(x))       # hidden layer 2 + ReLU activation
        x = self.fc3(x)               # output layer 
        return x
    

if __name__ == "__main__":
    # Prints a summary of the model
    mlp = MLP()
    summary(mlp, input_size=(1, 3, 100, 100))