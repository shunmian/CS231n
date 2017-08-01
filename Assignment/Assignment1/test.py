from __future__ import print_function
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.neural_net import TwoLayerNet
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()



scores = net.loss(X)

from cs231n.data_utils import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
input_size = 32 * 32 * 3
hidden_size = 80
num_classes = 10

learning_rates = [10e-3]
regs = [0.25]
results = {}
# Train the network
best_lr = None
best_rg = None
best_val_acc = 0
best_net = None

max_count = 100
from numpy.random import uniform
for count in range(max_count):
    reg = 10**uniform(-4,0)
    lr = 10**uniform(-3,-4)
    net = TwoLayerNet(input_size, hidden_size, num_classes)
    stats = net.train(X_train, y_train, X_val, y_val,
                          num_iters=2000, batch_size=1600,
                          learning_rate=lr, learning_rate_decay=0.95,
                          reg=reg, verbose=False)
    # Predict on the validation set

    val_acc = (net.predict(X_val) == y_val).mean()
    print('count: %d/%d lr: %e, rg: %e, Validation accuracy: %f' % (count,max_count,lr, reg, val_acc))
    results[(lr, reg)] = [stats, val_acc]
    if best_val_acc < val_acc:
        best_lr = lr
        best_rg = reg
        best_val_acc = val_acc
        best_net = net

for lr, reg in sorted(results):
    stats, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e val accuracy: %f' % (
        lr, reg, val_accuracy))

print('best validation accuracy achieved during cross-validation lr: %f, rg: %f, acc: %f' % (
best_lr, best_rg, best_val_acc))
