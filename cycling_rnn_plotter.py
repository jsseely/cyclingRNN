"""
  A script that runs summary analyses on a particular cluster run, and prints out .pdfs
  in the same folder.

  Usage:
    > python cycling_rnn_plotter.py RUNID
"""
import numpy as np
import scipy.io as sio
import sklearn as sk
from sklearn import decomposition

import matplotlib.pyplot as plt
import seaborn as sns

#import tensorflow as tf
import pandas as pd
#import datetime
from sklearn.grid_search import ParameterGrid

import pickle
import sys
import pprint


from cycling_rnn import *

# TODO: call matlab plotter
# TODO: make matlab plotter return actual values for the analyses
# e.g. in a mat file
# then write another function to scan through all analyses and get the 
# overall quality of the RNN ... 

# Color maps
cmap = sns.color_palette('RdBu', 5)[:2] + sns.color_palette('RdBu', 5)[-2:]
sns.set_palette(cmap)
sns.set_context('paper', font_scale=1.5)

# Load data.
pth = './saves/'+str(sys.argv[1])+'/'
param_grid = pickle.load(open(pth+'npsaves/param_grid.pickle', 'rb'))

# TODO: check if list of dicts, then do:
val_lengths = []
for p in range(len(param_grid)):
  val_lengths.append([len(v) for k, v in sorted(param_grid[p].items())])
# Else do:

# Number of sims in pth
total_sims = len(ParameterGrid(param_grid))

for sim, cur_params in enumerate(ParameterGrid(param_grid)):
  
  # load x and y data
  xpth = pth+'npsaves/'+str(sim)+'x.npy'
  ypth = pth+'npsaves/'+str(sim)+'y.npy'
  if not (os.path.exists(xpth) and os.path.exists(ypth)):
    continue
  x = np.load(xpth)
  y = np.load(ypth)

  print cur_params

  # TODO: fix time_axis save/restore...
  if cur_params['monkey']=='D':
    data = sio.loadmat('/Users/jeff/Documents/Python/_projects/cyclingRNN/drakeFeb_processed.mat')
    #time_axis = np.arange(2043, 4019, 25)
  else:
    data = sio.loadmat('/Users/jeff/Documents/Python/_projects/cyclingRNN/cousFeb_processed.mat')
    #time_axis = np.arange(2094, 4220, 25)

  emg = data['EMG']
  m1 = data['M1']

  x = x[:emg.shape[0], :emg.shape[1], :]
  y = y[:emg.shape[0], :emg.shape[1], :]

  # Plot 1: emg fits
  rows = 6
  cols = 2
  with sns.color_palette(n_colors=4):
    f, ax = plt.subplots(rows, cols,  figsize=(14, 12), sharex=True, sharey=False,
                         subplot_kw={'xticklabels':[], 'yticklabels':[]},
                         gridspec_kw={'wspace':0.01, 'hspace':0.01})
    for i in range(rows):
      for j in range(cols):
        muscle = np.ravel_multi_index((i, j), (rows, cols))
        ax[i, j].plot(emg[:, :, muscle], linewidth=1.5, alpha=0.8)
        ax[i, j].plot(y[:, :, muscle], '--', linewidth=3, alpha=1)
  f.suptitle(pprint.pformat(cur_params), fontsize=14, x=0.12, y=0.92,
             verticalalignment='bottom', horizontalalignment='left')
  f.savefig(pth+str(sim)+'fit.pdf', bbox_inches='tight', pad_inches=0.2) # TODO: proper formatting on str(i)
  f.clf()

  # Plot 2: RNN PCs
  pca_x = sk.decomposition.PCA(n_components=5)
  pca_x.fit(np.concatenate([x[:, ii, :] for ii in range(4)]))
  f = make_pairgrid([np.dot(x[:, ii, :], pca_x.components_.T) for ii in range(4)])
  f.fig.suptitle(pprint.pformat(cur_params), fontsize=14, x=0.05, y=1.0,
                            verticalalignment='bottom', horizontalalignment='left') 
  f.fig.savefig(pth+str(sim)+'pca.pdf', pad_inches=0.2)
  f.fig.clf()

  # Plot 3: Eigenvalues
  tf.reset_default_graph()
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(pth+'tfsaves/'+str(sim)+'.meta')
    new_saver.restore(sess, pth+'tfsaves/'+str(sim))
    Mat = sess.run([v for v in tf.all_variables() if v.name == 'RNN/BasicRNNCellNoise/Linear/Matrix:0'])[0]
    input_dims = Mat.shape[0]-Mat.shape[1]
    A = Mat[input_dims:]
    B = Mat[:input_dims]
  f = plot_eigs(A)
  f.suptitle(pprint.pformat(cur_params), fontsize=14, x=0.12, y=0.92,
             verticalalignment='bottom', horizontalalignment='left')
  f.savefig(pth+str(sim)+'eig.pdf', bbox_inches='tight', pad_inches=0.2)
  f.clf()

  # Plot 4: RNN activations
  rows = 3
  cols = 3
  f, ax = plt.subplots(rows,cols, figsize=(14, 8), sharex=True, sharey=True,
                       subplot_kw={'ylim':[-1, 1], 'xticklabels':[], 'yticklabels':[]},
                       gridspec_kw={'wspace':0.01, 'hspace':0.01})
  for i in range(rows):
    for j in range(cols):
      neuron = np.ravel_multi_index((i, j), (rows, cols))
      ax[i, j].plot(x[:, :, neuron])
  f.suptitle(pprint.pformat(cur_params), fontsize=14, x=0.12, y=0.92,
             verticalalignment='bottom', horizontalalignment='left')
  f.savefig(pth+str(sim)+'act.pdf', bbox_inches='tight', pad_inches=0.2)
  plt.clf()

  # Plot 5: Curvature:
  total_points = 11
  deg = 4
  k_m1 = 4*[None]
  k_emg = 4*[None]
  k_x = 4*[None]
  for c in range(4):
    k_m1[c], _ = get_generalized_curvature(m1[:, c, :], total_points, deg)
    k_emg[c], _ = get_generalized_curvature(emg[:, c, :], total_points, deg)
    k_x[c], _ = get_generalized_curvature(x[:, c, :], total_points, deg)

  with sns.color_palette('Set1', 3):
    f, ax = plt.subplots(2,2, figsize=(14, 8), sharex=True, sharey=True,
                         subplot_kw={'ylim':[0, 6], 'xticklabels':[], 'yticklabels':[]},
                         gridspec_kw={'wspace':0.01, 'hspace':0.01})
    ax[0, 0].plot(k_m1[0][:, 0])
    ax[0, 0].plot(k_x[0][:, 0])
    ax[0, 0].plot(k_emg[0][:, 0])
    ax[0, 0].plot(k_m1[0][:, 1], '--', linewidth=1, alpha=0.8)
    ax[0, 0].plot(k_x[0][:, 1], '--', linewidth=1, alpha=0.8)
    ax[0, 0].plot(k_emg[0][:, 1], '--', linewidth=1, alpha=0.8)
    ax[0, 0].set_title('top forward', y=0.9)
    
    ax[0, 1].plot(k_m1[1][:, 0])
    ax[0, 1].plot(k_x[1][:, 0])
    ax[0, 1].plot(k_emg[1][:, 0])
    ax[0, 1].plot(k_m1[1][:, 1], '--', linewidth=1, alpha=0.8)
    ax[0, 1].plot(k_x[1][:, 1], '--', linewidth=1, alpha=0.8)
    ax[0, 1].plot(k_emg[1][:, 1], '--', linewidth=1, alpha=0.8)
    ax[0, 1].set_title('bot forward', y=0.9)
    
    ax[1, 0].plot(k_m1[2][:, 0])
    ax[1, 0].plot(k_x[2][:, 0])
    ax[1, 0].plot(k_emg[2][:, 0])
    ax[1, 0].plot(k_m1[2][:, 1], '--', linewidth=1, alpha=0.8)
    ax[1, 0].plot(k_x[2][:, 1], '--', linewidth=1, alpha=0.8)
    ax[1, 0].plot(k_emg[2][:, 1], '--', linewidth=1, alpha=0.8)
    ax[1, 0].set_title('top backward', y=0.9)

    ax[1, 1].plot(k_m1[3][:, 0])
    ax[1, 1].plot(k_x[3][:, 0])
    ax[1, 1].plot(k_emg[3][:, 0])
    ax[1, 1].plot(k_m1[3][:, 1], '--', linewidth=1, alpha=0.8)
    ax[1, 1].plot(k_x[3][:, 1], '--', linewidth=1, alpha=0.8)
    ax[1, 1].plot(k_emg[3][:, 1], '--', linewidth=1, alpha=0.8)
    ax[1, 1].set_title('bot backward', y=0.9)
    ax[1, 1].legend(iter(ax[1,1].get_children()[:3]), ('M1', 'RNN', 'EMG'))

    f.suptitle(pprint.pformat(cur_params), fontsize=14, x=0.12, y=0.92,
               verticalalignment='bottom', horizontalalignment='left')
    f.savefig(pth+str(sim)+'curv.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close('all')
