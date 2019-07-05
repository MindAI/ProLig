# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

def get_predicted_labels(labels, logits, name='predicted_labels'):
  """Return prediced_labels (dtype: int32)"""
  # logits are of shape [batch_size, num_classes]
  # logits are unnormalized data (each row does not add up to 1)
  prediction_booling = tf.equal(labels, tf.cast(tf.argmax(logits, 1), tf.int32))
  predicted_labels = tf.cast(prediction_booling, tf.int32, name=name)
  return predicted_labels

def accuracy(labels, logits, name='accuracy'):
  prediction = tf.cast(get_predicted_labels(labels, logits), tf.float32)
  accuracy = tf.reduce_mean(prediction, name=name)
  return accuracy

def get_top_k_recall(labels, preds, k):
  """
    Notes:
      This code works for ties. e.g. If both top 1 and top 2 predictions 
      are same, this will not count top 2 correct predictions twice.
  """
  classes = np.unique(labels)
  top_k_recalls = []
  for elt in classes:
    labels_filter = (labels == elt)
    selection = preds[labels_filter][:,:k]
    top_k_recall = ((selection == elt).sum(axis=1) > 0).sum() / float(selection.shape[0])
    top_k_recalls.append(top_k_recall)
  return np.array(top_k_recalls)

def top_k_precision_recall(labels, preds, k):
  classes = np.unique(labels)
  top_k_precisions = []
  top_k_recalls = []
  for elt in classes:
    labels_filter = (labels == elt)

    # precision
    preds_filter = (preds[:,:k] == elt).sum(axis=1).astype(bool)
    top_k_precision = preds_filter[labels_filter].sum() / float(preds_filter.sum())
    top_k_precisions.append(top_k_precision)

    # recall
    selection = preds[labels_filter][:,:k]
    top_k_recall = (selection == elt).sum() / float(selection.shape[0])
    top_k_recalls.append(top_k_recall)
  return np.array(top_k_precisions), np.array(top_k_recalls)

def precision_recall_fscore(labels, preds):
  """preds: an array"""
  C = confusion_matrix(labels, preds)

  TPs = np.diag(C).astype(float)
  FPs = np.sum(C, axis=0) - TPs
  FNs = np.sum(C, axis=1) - TPs

  precisions = TPs/(TPs + FPs)
  precisions[np.isnan(precisions)] = 0 # set nan to 0

  recalls = TPs/(TPs + FNs)
  recalls[np.isnan(recalls)] = 0

  FScores = 2 * precisions * recalls / (precisions + recalls)
  FScores[np.isnan(FScores)] = 0
  
  return precisions, recalls, FScores, C

def full_evaluation(labels, preds, top_k=None, only_return_top_k_recalls=False):
  """
    Args:
      preds: either an array with shape = num_data, or a matrix with 
      shape = num_data * num_classes. 
      labels: an array shape = num_data
      top_k: a list containing the set of top k results to be returned.
          e.g. If returning results of top 1, 3, 5, then top_k = [1, 3, 5].
          If only returning top 1 results, then no need to specify top_k.
    Returns:
    Notes:
      This code works for ties. e.g. If both top 1 and top 2 predictions 
      are same, this will not count top 2 correct predictions twice.
    """
  if (top_k is None) | (top_k == [1]):
    if preds.ndim > 1:
      preds = preds[:,0]

    accuracy = (labels==preds).mean()
    precisions, recalls, FScores, ConfusionMatrix = precision_recall_fscore(
      labels, preds)

  elif len(top_k) > 1:
    assert preds.shape[1] > 1

    # This works for ties in preds. e.g. If both top 1 and top 2 predictions 
    # are same, this will not count top 2 correct predictions twice.
    accuracy = [ ( (np.equal(preds[:,:k], labels.reshape(-1,1)).sum(axis=1) > 0
      ).sum() / float(len(labels))) for k in top_k]

    if only_return_top_k_recalls:
      top_1_preds = preds[:,0]
      precisions, _, FScores, ConfusionMatrix = precision_recall_fscore(
        labels, top_1_preds)
      recalls = [get_top_k_recall(labels, preds, k) for k in top_k]
    else:
      ConfusionMatrix = None
      precisions = []
      recalls = []
      for k in top_k:
        top_k_precisions, top_k_recalls = top_k_precision_recall(labels, preds, k)
        precisions.append(top_k_precisions)
        recalls.append(top_k_recalls)
      
      FScores = [ (2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])) 
        for i in range(len(top_k))]

  return accuracy, precisions, recalls, FScores, ConfusionMatrix

def weighted_precision_recall_Fscore(labels, preds):
  """Args:
    preds: either an array with shape = num_data, or a matrix with 
    shape = num_data * num_classes.
    labels: an array shape = num_data
  """
  if preds.ndim > 1:
    preds = preds[:,0] # only retain the top 1 labels
  pr = precision_score(labels, preds, average='weighted')
  rc = recall_score(labels, preds, average='weighted')
  f_score = f1_score(labels, preds, average='weighted')
  return pr, rc, f_score


