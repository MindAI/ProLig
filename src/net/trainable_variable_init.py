# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

import tensorflow as tf
import numpy as np

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, weight_scale, reg, reg_type='l2', 
  reuse=None):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    weight_scale: standard deviation of a truncated Gaussian
    reg: add weight decay of type defined by reg_type multiplied by this float. 
        If None, weight decay is not added for this Variable.
    reg_type: default 'l2': L2 loss; 'l1': L1 loss
    reuse: a flag to control whether adding weight_decay. If initializing 
        variables for validation, then we do not need to add weight decay.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=weight_scale))
  if (reg is not None) & (reuse is None):
    if reg_type == 'l2':
      weight_decay = tf.mul(tf.nn.l2_loss(var), reg, name='L2_weight_loss')
    elif reg_type == 'l1':
      weight_decay = tf.mul(tf.reduce_sum(tf.abs(var)), reg, name='L1_weight_loss')
      # tf.reduce_sum: Computes the sum of elements across dimensions of a tensor
    tf.add_to_collection('losses', weight_decay)
  return var

def conv3d_layer_init(filter_size, num_filters, num_input_channels, reg, 
  reg_type='l2', weight_scale='AUTO', reuse=None):
  """reg: either None or a float32"""
  if weight_scale == 'AUTO':
    weight_scale = 1./np.sqrt(filter_size**3*num_filters/2.0)
  F = [filter_size, filter_size, filter_size, num_input_channels, num_filters]
  W = _variable_with_weight_decay('W', F, weight_scale, reg, reg_type=reg_type, 
    reuse=reuse)
  b = _variable_on_cpu('b', [num_filters], 
    tf.constant_initializer(0.0))
  return W, b

def fc_layer_init(input_dim, output_dim, reg, reg_type='l2', weight_scale='AUTO', 
  reuse=None):
  if weight_scale == 'AUTO':
    weight_scale = 1/np.sqrt(input_dim/2.0)
  W = _variable_with_weight_decay('W', [input_dim, output_dim], 
    weight_scale, reg, reg_type=reg_type, reuse=reuse)
  b = _variable_on_cpu('b', [output_dim], 
    tf.constant_initializer(0.0))
  return W, b
