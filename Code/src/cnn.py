"""Convolutional neural network for facial emotion recognition"""


import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation
from torchinfo import summary


col_chan = 3
out_classes = 7
class CNN(nn.Module):
   """Convolutional neural network with optional squeeze-and-excitation blocks.

    The network expects RGB images of shape (3, 100, 100) and outputs logits for
    7 emotion classes.

    Parameters
    ----------
    sq_ex : bool, optional
        If True, include squeeze-and-excitation blocks after convolution layers.
    dr_rate : float, optional
        Dropout rate applied before the final classification layer.
    """
   def __init__(self, sq_ex=False, dr_rate=0.5):
        super(CNN, self).__init__()

        self.sq_ex = sq_ex

        self.conv1 = nn.Conv2d(col_chan, 32, 10, padding=4, stride=3)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        if sq_ex:
            self.se1 = SqueezeExcitation(32, squeeze_channels=2)
            self.se2 = SqueezeExcitation(64, squeeze_channels=4)

        self.pool = nn.AvgPool2d(2, 2)

        self.fc1 = nn.Linear(64*16*16, 256)      # Hidden layer
        self.fc2 = nn.Linear(256, out_classes)   # Output layer
        self.dropout = nn.Dropout(dr_rate)

   def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.sq_ex:
            x = self.se1(x)
        x = F.relu(self.conv2(x))
        if self.sq_ex:
            x = self.se2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Prints summary of the CNN with and without SE-blocks
    cnn_plain = CNN(sq_ex=False).to("cpu")
    print(
        "\n\n" \
        "Plain CNN:\n" \
        "=========================================================================================="
        )
    summary(cnn_plain, input_size=(1, 3, 100, 100))

    cnn_sqex = CNN(sq_ex=True).to("cpu")
    print(
        "\n\n" \
        "CNN with Squeeze-and-Excitation blocks:\n" \
        "=========================================================================================="
        )
    summary(cnn_sqex, input_size=(1, 3, 100, 100))