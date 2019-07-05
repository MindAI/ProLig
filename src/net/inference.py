# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

from trainable_variable_init import *
from layers import *
from summary import *

import tensorflow as tf

def inference(Input, architecture, reg=None, reg_type='l2', only_reg_fc=False, 
  reuse=None, add_summary=False, mode='train'):
  """Input must be 5-D, with 1st dim = batch_size, 5th dim = num_channels. 
  If adding the same regularization to every weight, then specify reg; if also
  using L1 regularization, set reg_type='l1'.
  If only regularize fully-connected layers, then set only_reg_fc=True.  
  If called during testing, set mode='test'.
  If used on validation set after called for the training set, set reuse=True.

  return
    logits: shape=[batch_size, num_classes]"""

  conv_idx = 0; fc_idx = 0  # keep tracking of the number of conv, fc layers
  
  # keep tracking of the number of input channels for potential conv layers
  if len(Input.get_shape()) == 4:  # only 1 channel
    num_input_channels = 1
  elif len(Input.get_shape()) == 5:
    num_input_channels = Input.get_shape()[-1].value

  for layer in architecture:
    layer_type = layer[0]; params = layer[1]

    if layer_type=='conv':

      conv_idx += 1

      filter_size = params['filter_size']
      num_filters = params['num_filters']
      stride = params['stride']
      padding = params.get('padding', 'VALID')

      if params.get('reg'):
        reg = params.get('reg')
      if params.get('reg_type'):
        reg_type = params.get('reg_type')

      if only_reg_fc:
        reg = None
     
      with tf.variable_scope('conv'+str(conv_idx), reuse=reuse) as scope:
        W, b = conv3d_layer_init(filter_size, num_filters, 
          num_input_channels, reg, reg_type=reg_type, reuse=reuse)
        Input = conv3d_layer(Input, W, b, stride, padding=padding)

      num_input_channels = num_filters
      
      if (reuse==None) & (add_summary==True): # add summary
        activation_summary(Input)

    elif layer_type=='pool':
      pooling_size = params['pooling_size']
      stride = params.get('stride', pooling_size)
      Input = maxpooling_3d_layer(Input, pooling_size, stride)

      if (reuse==None) & (add_summary==True): # add summary
        activation_summary(Input)

    elif layer_type=='conv2fc':
      batch_size = Input.get_shape()[0].value
      Input = convd_to_fc(Input, batch_size)

    elif layer_type=='fc':

      fc_idx += 1

      hidden_dim = params['hidden_dim']
      input_dim = Input.get_shape()[1].value

      if params.get('reg'):
        reg = params.get('reg')
      if params.get('reg_type'):
        reg_type = params.get('reg_type')

      with tf.variable_scope('fc'+str(fc_idx), reuse=reuse) as scope:
        W, b = fc_layer_init(input_dim, hidden_dim, reg, reg_type=reg_type, 
          reuse=reuse)
        # dropout
        keep_prob = None
        if params.get('dropout'):
          keep_prob = params.get('dropout')
        Input = fc_layer(Input, W, b, dropout=keep_prob, mode=mode)
      
      if (reuse==None) & (add_summary==True): # add summary
        activation_summary(Input)

    elif layer_type=='output':

      num_classes = params['num_classes']
      input_dim = Input.get_shape()[1].value

      if params.get('reg'):
        reg = params.get('reg')
      if params.get('reg_type'):
        reg_type = params.get('reg_type')

      with tf.variable_scope('output', reuse=reuse) as scope:
        W, b = fc_layer_init(input_dim, num_classes, reg, reg_type=reg_type, 
          reuse=reuse)
        logits = fc_layer(Input, W, b, activation='None')
      
      if (reuse==None) & (add_summary==True): # add summary
        activation_summary(logits)

  return logits


      

