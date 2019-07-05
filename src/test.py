# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

from evaluate_test import evaluate_test
from net.evaluation import full_evaluation
from scipy.cluster.hierarchy import linkage, fcluster

import numpy as np
import pandas as pd
import os

def get_num_training_data_per_class(data_dir):
  trainIDs_fname = data_dir + 'IDs/trainIDs.list'
  all_IDs = open(trainIDs_fname).read().split('\n')[:-1]
  p1_ID_counts = [ID.split('_')[0] for ID in all_IDs if ID[-3:] == '_p1']
  ligands, counts = np.unique(np.array(p1_ID_counts), return_counts=True)
  return ligands.tolist(), counts

def get_ligand_similarity_linkage():
  ligand_sim_fname = '/net/kihara/home/zhu457/Proj/pdb_rest/ligand_comparison/' + \
    'all_ligands_pairwise_average_similarity.txt'

  with open(ligand_sim_fname) as f:
    entries = f.read().split('\n')[:-1]

  similarity_scores = [float(x.split('\t')[5]) for x in entries]
  # convert similarity_scores into distance vector
  dist_vec = 1 - np.array(similarity_scores)
  Z = linkage(dist_vec, method='average')
  return Z

def one_test(data_dir='', architectureID='000', image_shape=[25, 25, 25, 2], 
  step=1000, batch_size=128, 
  use_moving_average=True, use_checkpoint=True, 
  top_k = [1, 3, 5, 10, 15, 20, 25]):

  checkpoint_fname_prefix = '../checkpoints/' + data_dir.split('/')[-2] + \
    '_' + architectureID + '_'

  checkpoint_fname = checkpoint_fname_prefix + str(step) + '_steps.chk'

  labels, pred_labels_in_rank, _ = evaluate_test(
    architectureID, data_dir, batch_size, image_shape=image_shape, 
    use_checkpoint=use_checkpoint, checkpoint_fname=checkpoint_fname, 
    use_moving_average=use_moving_average)

  accuracy, precisions, recalls, FScores, ConfusionMatrix = full_evaluation(
    labels, pred_labels_in_rank, top_k=top_k, only_return_top_k_recalls=True)

  for i in xrange(len(top_k)):
    print 'top-%d accuracies:' % top_k[i], accuracy[i]
  print 'mean F1-score:', FScores.mean()
  print 'mean top-k recalls:', np.array(recalls).mean(axis=1)

  # save results to dataframe
  columns = ['type', 'numClasses']
  columns += ['top-' + str(k) + ' accuracy' for k in top_k]
  columns += ['mean top-' + str(k) + ' recall' for k in top_k]
  columns += ['mean_F1_score']
  d1 = pd.DataFrame(columns=columns)
  d1['type'] = ['all']
  d1['numClasses'] = len(np.unique(labels))
  d1[['top-' + str(k) + ' accuracy' for k in top_k]] = accuracy
  d1[['mean top-' + str(k) + ' recall' for k in top_k]] = \
    np.array(recalls).mean(axis=1)
  d1['mean_F1_score'] = FScores.mean()

  ligands, ligand_counts = get_num_training_data_per_class(data_dir)
  # ligand_counts: an array with length = number of ligand classes
  label_counts = ligand_counts[labels]
  # label_counts: an array of len(labels) that has number_of_data_in_class 
  # for each label

  # save per class results to dataframe
  columns = ['ligand', 'precision']
  columns += ['top-' + str(i) + ' recall' for i in top_k]
  columns += ['F-1 score']
  d2 = pd.DataFrame(columns=columns) 
  d2['ligand'] = ligands
  d2['F-1 score'] = FScores
  d2['precision'] = precisions
  d2[['top-' + str(i) + ' recall' for i in top_k]] = np.array(recalls).T

  # save d1 and d2 to one excel file
  if not os.path.exists('../results/'):
    os.mkdir('../results/')
  save_fname = '../results/results_' + architectureID + '_step_' + \
    str(step) + '.xlsx'
  writer = pd.ExcelWriter(save_fname)
  d1.to_excel(writer, 'Sheet1', index=False)
  d2.to_excel(writer, 'Sheet2', index=False)
  writer.save()


if __name__=='__main__':
  one_test(data_dir = '../data_151classes/', 
    architectureID='001', 
    step=150000)
