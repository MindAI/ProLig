# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

from train import train


if __name__=='__main__':
  train(data_dir = '../data_151classes/', 
    image_shape=[25, 25, 25, 2], 
    architectureID = '001', 
    reg = 4.0799890926421362e-06, 
    initial_learning_rate = 0.0004135724724575849, decay_steps=1000, 
    staircase=True, 
    add_summary = True, 
    print_and_write_summary_every=20, 
    save_checkpoint = True, 
    chk_very=1000, 
    max_step=200000)
