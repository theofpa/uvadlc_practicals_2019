"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch import nn
from torch.autograd import Variable
import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '800,400,200,100'
LEARNING_RATE_DEFAULT = 0.03
MAX_STEPS_DEFAULT = 5000
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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
  Performs training and evaluation of MLP model. 

  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []
  print("arch: ", dnn_hidden_units)

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  device = torch.device("cuda")

  dataset = cifar10_utils.get_cifar10()
  training = dataset['train']
  test = dataset['test']

  test_images = Variable(torch.tensor(test.images.reshape(test.images.shape[0], -1)))
  test_labels = torch.tensor(test.labels)

  model = MLP(n_inputs=32 * 32 * 3, n_hidden=dnn_hidden_units, n_classes=10).to(device)

  opt = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
  ce = nn.CrossEntropyLoss()

  test_accuracy= []
  train_accuracy = []
  loss_list = []

  for epoch in range(FLAGS.max_steps):

    x, y = training.next_batch(FLAGS.batch_size)
    x = Variable(torch.tensor(x).to(device))
    y = Variable(torch.tensor(y).to(device))

    opt.zero_grad()
    out = model.forward(x.reshape(FLAGS.batch_size, -1))
    loss = ce(out, y.max(1)[1])
    loss_list.append(float(loss))
    loss.backward()
    opt.step()

    if not epoch % FLAGS.eval_freq:
      train_accuracy.append(accuracy(out, y))
      out = model.forward(test_images.to(device))
      test_accuracy.append(accuracy(out, test_labels.to(device)))
      print('Epoch: ', epoch, 'Loss: ', loss, 'Accuracy: ', train_accuracy[-1], 'Test ac.:', test_accuracy[-1])

  out = model.forward(test_images.to(device))
  print('Test accuracy: ', accuracy(out, test_labels.to(device)))

  import seaborn as sns
  import matplotlib.pyplot as plt
  f, axes = plt.subplots(1, 2)
  ax=sns.lineplot(np.arange(0, MAX_STEPS_DEFAULT, EVAL_FREQ_DEFAULT), train_accuracy, ax=axes[0])
  ax=sns.lineplot(np.arange(0, MAX_STEPS_DEFAULT, EVAL_FREQ_DEFAULT), test_accuracy, ax=axes[0])
  ax.set_title('Training and test accuracy')
  ax.legend(['training','test'])
  ax=sns.lineplot(np.arange(0, MAX_STEPS_DEFAULT, 1), loss_list, ax=axes[1])
  ax.set_title('Loss')
  figure=ax.get_figure()
  figure.savefig("mlp-pytorch-results")

  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags(FLAGS):
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

#@ex.automain
#def main(model_type, hidden_units, learning_rate, epochs, batch_size):
def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags(FLAGS)

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  return(train())

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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