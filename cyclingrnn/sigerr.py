# signal errors
import numpy as np
from sklearn import metrics
from dtw import dtw

def r2_sigerr(x, y):
  """ R_squared of two signals"""
  err = metrics.r2_score(np.reshape(y, [-1, y.shape[-1]]),
                         np.reshape(x, [-1, x.shape[-1]]), multioutput='uniform_average')
  return err

def mse_sigerr(x, y):
  """ MSE of two signals """
  return np.mean((x - y)**2)

# Rpy2 stuff
def dtw_sigerr_R(x, y):
  """ dynamic time warping error (R package) """

  import rpy2.robjects.numpy2ri
  from rpy2.robjects.packages import importr
  rpy2.robjects.numpy2ri.activate()
  # Set up our R namespaces
  R = rpy2.robjects.r
  DTW = importr('dtw')

  dist = []
  idx1 = []
  idx2 = []
  for c in range(x.shape[1]):
    alignment = R.dtw(np.squeeze(x[:, c, :]), np.squeeze(y[:, c, :]), keep=False, window='none', step=DTW.asymmetric)
    dist.append(alignment.rx('distance')[0][0])
    idx1.append(np.array(alignment.rx('index1')[0]).astype('int')-1)
    idx2.append(np.array(alignment.rx('index2')[0]).astype('int')-1)
  #Now get R2
  x_ = []
  y_ = []
  for c in range(x.shape[1]):
    x_.append(x[path[c][0][5:-5], c, :]) # TODO: fix
    y_.append(y[path[c][1][5:-5], c, :])
  x_ = np.concatenate(x_)
  y_ = np.concatenate(y_)
  err = r2_sigerr(x_, y_)

  rpy2.robjects.numpy2ri.activate()

  return np.mean(dist), err

def dtw_sigerr_P(x, y):
  """ dynamic time warping error (python package) """
  def mydist(x, y):
    return np.linalg.norm(x-y)
  dist = []
  path = []
  for c in range(x.shape[1]):
    dist_, cost, acc, path_ = dtw(np.squeeze(x[:,c,:]), np.squeeze(y[:,c,:]), dist=mydist)
    dist.append(dist_)
    path.append(path_)
  
  x_ = []
  y_ = []
  for c in range(x.shape[1]):
    x_.append(x[path[c][0][10:-10], c, :])
    y_.append(y[path[c][1][10:-10], c, :])
  x_ = np.concatenate(x_)
  y_ = np.concatenate(y_)
  err = r2_sigerr(x_, y_)
  return np.mean(dist), err
