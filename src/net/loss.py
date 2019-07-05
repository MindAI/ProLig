# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

import tensorflow as tf

def mean_cross_entropy(logits, labels, weights_for_one_batch=None, 
  name='cross_entropy'):
  """
  Args:
    logits: Logits from inference(), un-normalized, shape = [batch_size, num_classes].
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    The mean of cross entropy, a scalar.
  """  

  if weights_for_one_batch is not None:
    cross_entropy = tf.contrib.losses.sparse_softmax_cross_entropy(logits, labels, 
      weights=weights_for_one_batch)
    # tf.contrib.losses.sparse_softmax_cross_entropy does not allow the arg 'name',
    # therefore cannot pass name='cross_entropy_per_example' to this function ...
  else:
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')    
  # shape: [batch_size]

  cross_entropy_mean = tf.reduce_mean(cross_entropy, name=name)
  # a scalar

  tf.add_to_collection('losses', cross_entropy_mean)

  return cross_entropy_mean

def get_total_loss():
  """Warning: This function must be called before the validation loss is added to 
  the "losses" collection.
  Compute the total training loss by adding all the loss terms in the "losses" 
  collection together, which include weight decays of all the layers and 
  the cross entropy loss"""
  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  # Before this operation, all of the weight decay losses have been added
  # to the collection losses, each being a scalar.
  # Now by applying tf.add_n, the cross_entropy_mean loss and weight_decay
  # loss are added together.
  # So the returned is a scalar
  tf.add_to_collection('losses', total_loss)
  return total_loss


def softmax_loss(train_logits, train_labels, weights_for_one_batch=None):
  """ A wrapper for the 2 functions above: mean_cross_entropy(), total_loss().
  Warning: Do not use this function on validation set.
  Args:
    train_logits: logits for the training set.
    train_labels: true labels for the training set.
  Returns:
    total_loss: a scalar"""
  cross_entropy_mean = mean_cross_entropy(train_logits, train_labels, 
    weights_for_one_batch=weights_for_one_batch)
  total_loss = get_total_loss()
  return total_loss
  
