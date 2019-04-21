import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    

    """
    super(CustomBatchNormAutograd, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # Save parameters for the number of neurons and epsilon.
    self.n_neurons = n_neurons
    self.eps = eps
    # Initialize parameters gamma and beta via nn.Parameter
    self.gamma = nn.Parameter(torch.ones(n_neurons))
    self.beta = nn.Parameter(torch.zeros(n_neurons))
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor

    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # Check for the correctness of the shape of the input tensor.
    assert input.shape[1] == self.n_neurons, "the shape of the input vector is not correct"

    # Implement batch normalization forward pass as given in the assignment.

    # compute mean
    cmean = input.mean(dim=0)
    # compute variance
    cvariance = input.var(dim=0, unbiased=False)
    # normalize with a constant eps << 1 to avoid numerical instability
    xnorm = (input - cmean) / torch.sqrt(cvariance + self.eps)
    # scale and fit
    out = torch.mul(self.gamma, xnorm) + self.beta

    assert input.shape == out.shape, "the output shape does not match the input shape"
    ########################
    # END OF YOUR CODE    #
    #######################

    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor


    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # input: input tensor of shape (n_batch, n_neurons)
    n_batch, n_neurons = input.shape

    # manually compute mean
    cmean = 1 / n_batch * torch.sum(input, dim=0)
    x_mean = (input - cmean.unsqueeze(0))

    # manually compute variance
    cvariance = 1 / n_batch * torch.einsum('ij,ij->j', (x_mean, x_mean))
    cvariance.unsqueeze_(0)

    # normalize with epsilon to avoid numerical instability
    norm = 1 / torch.sqrt(cvariance + eps)
    xnorm = x_mean * norm

    # scale and fit
    out = xnorm * gamma + beta

    # Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
    ctx.save_for_backward(norm, xnorm, gamma)

    # Store constant non-tensor objects via ctx.constant=myconstant
    ctx.n_batch = n_batch
    ctx.gamma = gamma

    ########################
    # END OF YOUR CODE    #
    #######################

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments

    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    # Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
    norm, xnorm, gamma = ctx.saved_tensors
    n_batch = ctx.n_batch


    # Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
    # inputs to None. This should be decided dynamically.
    grad_input = grad_gamma = grad_beta = None
    dx_norm = grad_output * gamma
    if ctx.needs_input_grad[0]:
      grad_input = 1/n_batch * norm * ((n_batch * dx_norm)-torch.sum(dx_norm, dim=0)-(xnorm * torch.sum(dx_norm * xnorm, dim=0)))
    if ctx.needs_input_grad[1]:
      grad_beta = torch.sum(grad_output, dim=0)
    if ctx.needs_input_grad[2]:
      grad_gamma = torch.sum(xnorm * grad_output, dim=0)

    ########################
    # END OF YOUR CODE    #
    #######################

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability

    """
    super(CustomBatchNormManualModule, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # Save parameters for the number of neurons and epsilon.
    self.n_neurons = n_neurons
    self.eps = eps
    # Initialize parameters gamma and beta via nn.Parameter
    self.gamma = nn.Parameter(torch.ones(n_neurons))
    self.beta = nn.Parameter(torch.zeros(n_neurons))
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor

    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    assert input.shape[1] == self.n_neurons , "the input tensor does not have the correct shape"
    layer = CustomBatchNormManualFunction()
    out = layer.apply(input, self.gamma, self.beta, self.eps)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
