"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn
from custom_batchnorm import CustomBatchNormManualModule
class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(MLP, self).__init__()
    self.layers = []
    in_features = n_inputs
    for out_features in n_hidden:
      linear = nn.Linear(in_features, out_features)
      ##batchnorm = nn.BatchNorm1d(out_features)
     # dropout = nn.Dropout(0.2)
      relu = nn.ReLU()
      self.layers.append(linear)
      ##self.layers.append(batchnorm)
      self.layers.append(CustomBatchNormManualModule(out_features))
     # self.layers.append(dropout)
      self.layers.append(relu)

      in_features = out_features
    #dropout = nn.Dropout()
    #self.layers.append(dropout)
    linear = nn.Linear(in_features, n_classes)
    softmax = nn.Softmax()
    self.layers.append(linear)
    # self.layers.append(softmax)
    self.sequential = nn.Sequential(*self.layers)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.sequential(x.reshape(x.shape[0], -1))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
