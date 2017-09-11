import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) # for every X, calculate its score agains each class
    correct_class_score = scores[y[i]] # y[i] means the correct class, scores[y[i]] means the predicted score for the correct class
    count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1, other classes predited score minus correct_class_score
      if margin > 0:
        loss += margin
        count += 1
        dW[:, j] += X[i,:].T
        dW[:, y[i]] -= X[i, :].T
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # average dW too
  dW /= num_train
    
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
    
  # Add regularization gradient to the dW
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)

  row_idx = np.arange(X.shape[0])
  
  # get correct_scores
  correct_scores = scores[row_idx, y]
  # calculate max margin
  margins = np.maximum(0, (scores - np.array([correct_scores]).T) + 1)
  # set correct_scores margin = 0
  margins[row_idx, y] = 0
  # row sum
  loss = np.sum(margins)

  # average loss
  loss /= X.shape[0]

  # add reg
  loss += reg*np.sum(np.sum(np.square(W), axis=1), axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # for every margin > 0, decrease dw_yi by x_i, increase dw_j by x_i
  margins[margins > 0] = 1
  margins[margins < 0] = 0
  margins[np.arange(0, X.shape[0]), y] = 0
  margins[np.arange(0, X.shape[0]), y] = -1 * np.sum(margins, axis=1)
  dW = X.T.dot(margins)
    
  # average dW
  dW /= X.shape[0]
    
  # Add regularization gradient to the dW
  dW += 2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
