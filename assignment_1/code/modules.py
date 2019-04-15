"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.params = {'weight': None, 'bias': None}
        self.grads = {'weight': None, 'bias': None}
        mu=0
        sigma=0.0001
        self.params['weight']=np.random.normal(mu,sigma,[in_features ,out_features])
        self.params['bias']=np.zeros(out_features)
        self.grads['weight']=np.zeros([in_features, out_features])
        self.grads['bias']=np.zeros(out_features)
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # LinearModule.foward: ~x = x W + b
        out = x @ self.params['weight'] + self.params['bias']
        # do the trick to store the input
        self.x = x
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # dL/dx = dL/dY * W.T
        dx = dout @ self.params['weight'].T
        # dL/dW = X.T * dL/dY, use the stored input
        self.grads['weight'] =self.x.T @ dout
        # dL/db = dL/dz * 1 = sum of lines of L
        self.grads['bias'] = np.sum(dout)
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx

class ReLUModule(object):
    """
    ReLU activation module.
    """
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # max(0,x)
        out=np.maximum(np.zeros(x.shape),x)
        # store for backprop
        self.x=x
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx = dout * (self.x > 0)
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx

class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        b=np.max(x)
        y=np.exp(x-b)
        out = y/np.sum(y, axis=-1, keepdims=True)
        self.x=x
        self.out=out
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        diagonial = np.einsum('ij,kj->ikj', self.out, np.eye(self.out.shape[1]))
        softmax = np.einsum('ij,il->ijl', self.out, self.out)
        dx = np.einsum('ij,ijk->ik', dout, diagonial - softmax)
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx

class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    def forward(self, x, y):
        """
        Forward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        Implement forward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out=np.sum(-np.einsum('ij,ij->j', np.log(x), y))/x.shape[0]
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dx=-1/x.shape[0]*(y/x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx
