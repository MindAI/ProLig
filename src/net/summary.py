# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

import tensorflow as tf
import re

# def variable_summaries(var):
#   """Attach a lot of summaries to a Tensor."""
#   name = var.op.name
#   with tf.name_scope('summaries'):
#     mean = tf.reduce_mean(var)
#     tf.summary.scalar('mean/' + name, mean)
#     with tf.name_scope('stddev'):
#       stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
#     tf.summary.scalar('sttdev/' + name, stddev)
#     tf.summary.scalar('max/' + name, tf.reduce_max(var))
#     tf.summary.scalar('min/' + name, tf.reduce_min(var))
#     tf.summary.histogram(name, var)

def activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor."""
  name = var.op.name
  mean = tf.reduce_mean(var)
  tf.summary.scalar(name + '/mean', mean)
  stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
  tf.summary.scalar(name + '/stddev', stddev)
  tf.summary.scalar(name + '/max', tf.reduce_max(var))
  tf.summary.scalar(name + '/min', tf.reduce_min(var))
  tf.summary.histogram(name + '/distribution', var)


def loss_summaries():
  """Add summaries for all loss terms in the collection "losses", including 
  the weight decay in each layer, the cross entropy loss, the total_loss, 
  and optionally the validation loss.
  Warning: call this function only after all the loss terms have been added 
  to the collection "losses".

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='loss_avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses)
  # losses is a list, with each item a scalar
  # losses + [total_loss] is also a list

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +'_raw_', l)
    tf.summary.scalar(l.op.name + '_movingAvg_', loss_averages.average(l))

  return loss_averages_op


def accuracy_summaries(accu):
  avg = tf.train.ExponentialMovingAverage(0.9, name='accu_avg')
  avg_op = avg.apply([accu])
  tf.summary.scalar(accu.op.name +'_raw_', accu)
  tf.summary.scalar(accu.op.name + '_movingAvg_', avg.average(accu))
  return avg_op

def raw_and_avg_summaries(var_or_varList, add_summary=False):
  avg = tf.train.ExponentialMovingAverage(0.9, name='avg')

  if isinstance(var_or_varList, list): # if var_or_varList is a list
    var_list = var_or_varList
  else:
    var_list = [var_or_varList]

  avg_op = avg.apply(var_list)

  var_avg_values = []
  for var in var_list:
    var_avg_value = avg.average(var)
    var_avg_values.append(var_avg_value)
    if add_summary:
      tf.summary.scalar(var.op.name + '_raw_', var)
      tf.summary.scalar(var.op.name + '_movingAvg_', var_avg_value)

  if isinstance(var_or_varList, list):
    return avg_op, var_avg_values
  else: # if var_or_varList is just a var, then only return the avg_value
    return avg_op, var_avg_values[0]


