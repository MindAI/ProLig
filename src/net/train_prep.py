# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

from net.summary import *

import tensorflow as tf

def build_train_op(total_loss, global_step, 
  initial_learning_rate=0.001, decay_steps=100, lr_decay_rate=0.99, staircase=False, 
  optimizer='Adam', 
  moving_average_decay=0.9999, 
  add_summary=False):
  """For now, optimizer can be chosen from 'Adam' and 'GradientDescent'. """

  decayed_learning_rate = tf.train.exponential_decay(
    initial_learning_rate,
    global_step,
    decay_steps,
    lr_decay_rate, 
    staircase=staircase)

  # add summary of changing learning rate
  if add_summary:
    tf.summary.scalar('learning_rate', decayed_learning_rate)

# build op
  if optimizer == 'Adam':
    opt = tf.train.AdamOptimizer(learning_rate=decayed_learning_rate)
  elif optimizer == 'GradientDescent':
    opt = tf.train.GradientDescentOptimizer(learning_rate=decayed_learning_rate)

  # Compute gradients.
  grads = opt.compute_gradients(total_loss)
  # grads is a list of tuples, each tuple includes the gradient 
  # and the corresponding variable

  # Apply gradients
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  if add_summary:
    for var in tf.trainable_variables():
      variable_summaries(var)

    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name+'/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
    moving_average_decay, global_step)
  variable_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op



  




  




