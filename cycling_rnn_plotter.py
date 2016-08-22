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

from cycling_rnn import *

## functions
def make_pairgrid(d):
  ''' in: (d1,d2,d3,d4) '''
  df = pd.DataFrame(np.concatenate(d))
  cond_labels = d[0].shape[0]*['fw top'] + d[1].shape[0]*['fw bot'] + d[2].shape[0]*['bw top'] + d[3].shape[0]*['bw bot']
  df['condition'] = cond_labels
  g = sns.PairGrid(df, hue='condition', diag_sharey=True)
  g.map_diag(plt.hist)
  g.map_offdiag(plt.plot)
  dmax = np.max(np.concatenate(d))
  g.add_legend()
  return g

def plot_eigs(A_mat):
  """
    Docstring
  """
  w, _ = np.linalg.eig(A_mat)
  re_w = np.real(w)
  im_w = np.imag(w)
  f = plt.figure(figsize=(5, 5))
  plt.plot(re_w, im_w, 'o', alpha=0.9)
  theta = np.linspace(0, 2*np.pi, num=50)
  x_cir = np.cos(theta)
  y_cir = np.sin(theta)
  plt.plot(x_cir, y_cir, 'k', linewidth=0.5, alpha=0.5)
  plt.plot([-100, 100], [0, 0], 'k', linewidth=0.5, alpha=0.5)
  plt.plot([0, 0], [-100, 100], 'k', linewidth=0.5, alpha=0.5)
  plt.xlim([-1.5, 1.5])
  plt.ylim([-1.5, 1.5])
  return f

# Load data.
RUN = '0822-1343-02'
pth = './saves/'+RUN+'/'
x_tf = np.load(pth+'npsaves/x.npy')
y_tf = np.load(pth+'npsaves/y.npy')
param_grid = pickle.load(open(pth+'npsaves/param_grid.pickle'))
print x_tf.shape
print param_grid

# Color maps
cmap = sns.color_palette('RdBu', 5)[:2] + sns.color_palette('RdBu', 5)[-2:]
sns.set_palette(cmap)
sns.set_context('paper', font_scale=1.5)

for i_, cur_params in enumerate(ParameterGrid(param_grid)):
  print cur_params
  param_inds = np.unravel_index(i_, x_tf.shape) # TODO: 'C' or 'F'? Pretty sure this is right.

  # Inefficient loading of data each iteration.
  if cur_params['monkey'] == 'D':
    data = sio.loadmat('/Users/jeff/Documents/Python/_projects/cyclingRNN/drakeFeb.mat')
  else:
    data = sio.loadmat('/Users/jeff/Documents/Python/_projects/cyclingRNN/cousFeb.mat')

  # Preprocess data
  emg = preprocess_array(data['D'][0, 0]['EMG'])
  m1 = preprocess_array(data['D'][0, 0]['M1'], alpha=5)

  time_axis, time_inds1, time_inds2 = get_time_axis(data['D'][0, 0]['KIN'])

  emg = emg[time_axis]
  m1 = m1[time_axis]

  # down-select data
  for ii in range(x_tf.size):
    param_inds = np.unravel_index(ii, x_tf.shape)
    x_tf[param_inds] = x_tf[param_inds][:emg.shape[0], :emg.shape[1], :]
    y_tf[param_inds] = y_tf[param_inds][:emg.shape[0], :emg.shape[1], :]

  # Plot 1: emg fits
  rows = 4
  cols = 2
  with sns.color_palette(n_colors=4):
    f, ax = plt.subplots(rows, cols, figsize=(20, 20))
    for i in range(rows):
      for j in range(cols):
        muscle = np.ravel_multi_index((i, j), (rows, cols))
        ax[i, j].plot(emg[:, :, muscle], linewidth=1.5, alpha=0.8)
        ax[i, j].plot(y_tf[param_inds][:, :, muscle], '--', linewidth=3, alpha=1)
        ax[i, j].set_title(str(muscle))
  f.suptitle(cur_params, fontsize=16)
  f.savefig(pth+str(i_)+'fit.pdf') # TODO: proper formatting on str(i)
  f.clf()

  # Plot 2: RNN PCs
  pca_x = sk.decomposition.PCA(n_components=5)
  pca_x.fit(np.concatenate([x_tf[param_inds][:, ii, :] for ii in range(4)]))
  # plot PCs for the RNN state variable
  f = make_pairgrid([np.dot(x_tf[param_inds][:, ii, :], pca_x.components_.T) for ii in range(4)])
  f.fig.suptitle(cur_params, fontsize=16)
  f.fig.savefig(pth+str(i_)+'pca.pdf')
  f.fig.clf()

  # Plot 3: Eigenvalues
  tf.reset_default_graph()
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(pth+'tfsaves/'+str(i_)+'.meta')
    new_saver.restore(sess, pth+'tfsaves/'+str(i_))
    Mat = sess.run([v for v in tf.all_variables() if v.name == 'RNN/BasicRNNCell/Linear/Matrix:0'])[0]
    input_dims = Mat.shape[0]-Mat.shape[1]
    A = Mat[input_dims:]
    B = Mat[:input_dims]
  f = plot_eigs(A)
  f.savefig(pth+str(i_)+'eig.pdf')
  f.clf()

  # Plot 4: RNN activations
  rows = 10
  cols = 4
  f, ax = plt.subplots(rows, cols, figsize=(20, 20))
  for i in range(rows):
    for j in range(cols):
      neuron = np.ravel_multi_index((i, j), (rows, cols))
      ax[i, j].plot(x_tf[param_inds][:, :, neuron])
      ax[i, j].set_title(str(neuron))
  f.suptitle(cur_params, fontsize=16)
  f.savefig(pth+str(i_)+'act.pdf')
  plt.close('all')

