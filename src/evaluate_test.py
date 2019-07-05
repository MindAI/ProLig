# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

from net.net_struct import get_architecture
from net.inference import inference
from net.evaluation import full_evaluation, weighted_precision_recall_Fscore

import tensorflow as tf
import glob
import h5py
import numpy as np

def get_batch_logits(architecture, batch_size, reuse=None, 
  image_shape=[25, 25, 25, 2]):
  X = tf.placeholder(tf.float32, shape = [batch_size] + image_shape)
  logits = inference(X, architecture, reuse=reuse, mode='test')
  softmax_logits = tf.nn.softmax(logits)
  return X, logits, softmax_logits
  
def evaluate_one_test_file(architecture, test_file_name, batch_size, logits_all, 
  softmax_logits_all, sess, X, logits, softmax_logits, 
  image_shape=[25, 25, 25, 2]):
  hf = h5py.File(test_file_name, 'r')
  data = hf.get('data')
  labels = np.array(hf.get('labels'))

  for i in xrange(0, data.shape[0]-batch_size, batch_size):
    logits_value, softmax_logits_value = sess.run([logits, softmax_logits], 
      feed_dict={X: data[i:(i+batch_size)]})
    logits_all.append(logits_value)
    softmax_logits_all.append(softmax_logits_value)
  # if data.shape[0]-batch_size <= 0, then i will not be assigned

  if data.shape[0]-batch_size <= 0:
    i = - batch_size

  last_batch_size = data.shape[0] % batch_size
  if last_batch_size != 0:
    last_X, last_logits, last_softmax_logits = get_batch_logits(architecture, 
      last_batch_size, reuse=True, image_shape=image_shape)
    last_logits_value, last_softmax_logits_value = sess.run(
      [last_logits, last_softmax_logits], 
      feed_dict={last_X: data[(i+batch_size):]})    
  else:
    last_logits_value, last_softmax_logits_value = sess.run(
      [logits, softmax_logits], 
      feed_dict={X: data[(i+batch_size):]})

  logits_all.append(last_logits_value)
  softmax_logits_all.append(last_softmax_logits_value)

  hf.close()

  return logits_all, softmax_logits_all, labels


def evaluate_test(architectureID, data_dir, batch_size, image_shape=[25, 25, 25, 2], 
  use_checkpoint=True, checkpoint_fname=None, sess=None,  
  use_moving_average=False, moving_average_decay=0.9999, 
  ):
  architecture = get_architecture(architectureID)

  if use_checkpoint:
    assert checkpoint_fname
    reuse = None
  else:
    assert sess is not None
    reuse = True

  X, logits, softmax_logits = get_batch_logits(architecture, batch_size, 
    reuse=reuse, image_shape=image_shape)

  if use_moving_average:
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
  else:
    saver = tf.train.Saver()

  if use_checkpoint:
    sess = tf.Session()

  saver.restore(sess, checkpoint_fname)

  logits_all = []
  softmax_logits_all = []
  labels = []
  for fname in glob.glob(data_dir+'test*.h5'):
    logits_all, softmax_logits_all, this_labels = evaluate_one_test_file(
      architecture, fname, batch_size, logits_all, softmax_logits_all, sess, 
      X, logits, softmax_logits, image_shape=image_shape)
    labels.append(this_labels)

  sess.close()

  labels = np.concatenate(labels, axis=0)
  logits_all = np.concatenate(logits_all, axis=0)
  softmax_logits_all = np.concatenate(softmax_logits_all, axis=0)

  pred_labels_in_rank = np.fliplr(np.argsort(logits_all))
  # shape = num_test * num_classes
  # each row, the most possible label is in the first column (the first number), 
  # the 2nd most in the 2nd column, etc.

  return labels, pred_labels_in_rank, softmax_logits_all




