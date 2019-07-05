# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

import tensorflow as tf

def conv3d_layer(X, W, b, stride, padding='VALID', activation='ReLU'):  
  S = [1, stride, stride, stride, 1]
  convd = tf.nn.conv3d(X, W, S, padding)
  convd = tf.nn.bias_add(convd, b, name='conv')
  if activation=='ReLU':
    convd = tf.nn.relu(convd, name='conv_relu')
  return convd

def maxpooling_3d_layer(X, pooling_size, stride, padding='VALID'):
  ksize = [1, pooling_size, pooling_size, pooling_size, 1]
  strides = [1, stride, stride, stride, 1]
  return tf.nn.max_pool3d(X, ksize, strides, padding)

def convd_to_fc(X, batch_size):
  return tf.reshape(X, [batch_size, -1])

def fc_layer(X, W, b, activation='ReLU', dropout=None, mode='train'):
  """If using dropout, set dropout = keep_prob. 
  choice of mode: {'train', 'test'}"""
  fcd = tf.nn.bias_add(tf.matmul(X, W), b, name='fc')
  if activation=='ReLU':
    fcd = tf.nn.relu(fcd, 'fc_relu')
  if (dropout is not None) & (mode=='train'):
    fcd = tf.nn.dropout(fcd, dropout, name='fc_relu_dropout')
  return fcd
