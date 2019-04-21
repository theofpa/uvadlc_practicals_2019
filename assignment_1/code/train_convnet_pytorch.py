"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  [batch_size, n_classes] = targets.shape
  accuracy = float(torch.sum(targets[np.arange(0, batch_size), torch.argmax(predictions, dim=1)] == 1))/batch_size
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  device = torch.device("cuda")

  cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

  dataset = cifar10_utils.get_cifar10()
  training = dataset['train']
  test = dataset['test']

  model = ConvNet(3, 10)
  model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

  ce = torch.nn.CrossEntropyLoss()

  test_accuracy= []
  loss_list = []

  for epoch in np.arange(0, FLAGS.max_steps):
    x, y = training.next_batch(FLAGS.batch_size)
    x = Variable(torch.tensor(x).to(device))
    y = Variable(torch.tensor(y).to(device))

    optimizer.zero_grad()
    model.train()
    yh = model.forward(x)
    loss = ce(yh, torch.max(y,1)[1])
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()

    if (epoch + 1) % int(FLAGS.eval_freq) == 0:

      acc = []
      with torch.no_grad():
        for _ in np.arange(0, (test.num_examples // FLAGS.batch_size)):
          x, y = test.next_batch(FLAGS.batch_size)
          x = torch.tensor(x).to(device)
          y = torch.tensor(y).to(device)

          model.eval()
          yh = model.forward(x)
          acc.append(accuracy(yh, y))
        test_accuracy.append(np.mean(acc))
        print(np.mean(acc))

  import seaborn as sns
  import matplotlib.pyplot as plt
  f, axes = plt.subplots(1, 2)
  ax=sns.lineplot(np.arange(0, MAX_STEPS_DEFAULT, EVAL_FREQ_DEFAULT), test_accuracy, ax=axes[0])
  ax.set_title('Test accuracy')
  ax=sns.lineplot(np.arange(0, MAX_STEPS_DEFAULT, 1), loss_list, ax=axes[1])
  ax.set_title('Loss')
  figure=ax.get_figure()
  figure.savefig("cnn-pytorch-results")

########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()