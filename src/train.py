# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

from net.net_struct import get_architecture
from util.read_input import get_train_batch, get_val_batch
from net.inference import inference
from net.loss import softmax_loss, mean_cross_entropy
from net.train_prep import build_train_op
from net.evaluation import accuracy
from net.summary import *

import tensorflow as tf
import numpy as np
import os


def train(data_dir='', image_shape=[25, 25, 25, 2], batch_size=128, 
  num_train_examples=None, evaluate_validation=False, 
  num_val_examples=None, validation_batch_size=1000, 
  architectureID=None, 
  reg=None, reg_type='l2', only_reg_fc=False, 
  initial_learning_rate=0.001, decay_steps=100, lr_decay_rate=0.99, staircase=False,
  optimizer='Adam', 
  moving_average_decay=0.9999, 
  add_summary=False, print_and_write_summary_every=100, 
  save_checkpoint=False, chk_very=1000, 
  max_step=100000):

  global_step = tf.Variable(0, trainable=False, name='global_step')

  # Get image batch and label batch
  X_train_batch, y_train_batch = get_train_batch(data_dir, image_shape=image_shape, 
    batch_size=batch_size, num_examples=num_train_examples)

  architecture = get_architecture(architectureID)

  logits = inference(X_train_batch, architecture, reg=reg, reg_type=reg_type, 
    only_reg_fc=only_reg_fc, add_summary=add_summary) 
  # shape = [batch_size, num_classes]

  total_loss = softmax_loss(logits, y_train_batch)  # a scalar

  # Compute training accuracy 
  train_accu = accuracy(y_train_batch, logits, name='training_accuracy')

  # On validation set
  if evaluate_validation:
    # Get data
    X_val_batch, y_val_batch = get_val_batch(data_dir, image_shape=image_shape, 
      batch_size=validation_batch_size, num_examples=num_val_examples)
    val_logits = inference(X_val_batch, architecture, reg=reg, reg_type=reg_type, 
      only_reg_fc=only_reg_fc, add_summary=add_summary, reuse=True, mode='test')
    # Compute validation accuracy
    val_accu = accuracy(y_val_batch, val_logits, name='validation_accuracy')
    # compute loss for validation
    val_cross_entropy_mean = mean_cross_entropy(val_logits, y_val_batch, 
      name='validation_cross_entropy')

  loss_averages_op, loss_avg_values = raw_and_avg_summaries(
    tf.get_collection('losses'), add_summary=add_summary)

  total_loss_avg_value = tf.get_default_graph().get_tensor_by_name(
    'total_loss/avg:0')

  if evaluate_validation:
    val_loss_avg_value = tf.get_default_graph().get_tensor_by_name(
      'validation_cross_entropy/avg:0')

  train_accu_avg_op, train_accu_avg_value = raw_and_avg_summaries(
    train_accu, add_summary=add_summary)

  val_accu_avg_op = None
  if evaluate_validation:
    val_accu_avg_op, val_accu_avg_value = raw_and_avg_summaries(
      val_accu, add_summary=add_summary)

  # Build a Graph that trains the model
  control_dependency_ops = [loss_averages_op, train_accu_avg_op]
  if evaluate_validation:
    control_dependency_ops += [val_accu_avg_op]
  
  with tf.control_dependencies(control_dependency_ops):
    train_op = build_train_op(total_loss, global_step, 
      initial_learning_rate=initial_learning_rate, decay_steps=decay_steps, 
      lr_decay_rate=lr_decay_rate, staircase=staircase, 
      optimizer=optimizer, 
      moving_average_decay=moving_average_decay, 
      add_summary=add_summary)

  # Create a saver.
  if save_checkpoint:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
    checkpoint_fname_prefix = '../checkpoints/' + data_dir.split('/')[-2] + \
      '_' + architectureID + '_'
    if not os.path.exists('../checkpoints/'):
      os.mkdir('../checkpoints/')

  # Build the summary operation based on the TF collection of Summaries.
  if add_summary:
    summary_op = tf.summary.merge_all()

  # Build an initialization operation to run below.
  init = tf.global_variables_initializer()

  # Start running operations on the Graph.
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
  sess.run(init)

  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)

  if add_summary:
    summary_save_dir = '../summaries/'
    if not os.path.exists(summary_save_dir):
      os.mkdir(summary_save_dir)

    summary_writer = tf.summary.FileWriter(summary_save_dir, sess.graph)


  for step in xrange(1, max_step+1):
    _, V_loss, V_loss_avg, V_train_accu, V_train_accu_avg = sess.run([train_op, 
      total_loss, total_loss_avg_value, 
      train_accu, train_accu_avg_value])

    assert not np.isnan(V_loss), 'Model diverged with loss = NaN'

    if step % print_and_write_summary_every == 0:

      if evaluate_validation:
        V_val_accu, V_val_accu_avg = sess.run([val_accu, val_accu_avg_value])
        print 'Step %d, total_loss = %.4f, total_loss_avg = %.4f, train_accu = %.4f, train_accu_avg = %.4f, val_accu = %.4f, val_accu_avg = %.4f' % (
          step, V_loss, V_loss_avg, V_train_accu, V_train_accu_avg, 
            V_val_accu, V_val_accu_avg)
      else:
        print 'Step %d, total_loss = %.4f, total_loss_avg = %.4f, train_accu = %.4f, train_accu_avg = %.4f' % (
          step, V_loss, V_loss_avg, V_train_accu, V_train_accu_avg)        

      if add_summary:
        summaries = sess.run(summary_op)
        summary_writer.add_summary(summaries, step)

    if (save_checkpoint) & (step % chk_very == 0):
      checkpoint_fname = checkpoint_fname_prefix + str(step) +'_steps.chk'
      saver.save(sess, checkpoint_fname)

