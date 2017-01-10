

def tangling(signal_in, alpha=0.1, dt=0.025):
  '''
    Computes "tangling," a global-ish measure of curvature of a curve
    Inputs
      signal_in: (T, batch_size, n) signal
      alpha: factor for how much to scale np.var() in the denominator
      dt: time units
    Outputs
      Q: (T, batch_size) tangling metric
  '''
  sig_diff = np.diff(signal_in, axis=0)
  signal_in = signal_in[1:, :, :]
  Q = np.zeros(signal_in.shape[:2])
  const = alpha*np.sum(np.var(signal_in, axis=(0,1))) # var across all data, or var across one axis then sum?
  for t in range(signal_in.shape[0]):
    for b in range(signal_in.shape[1]):
      num = np.sum((sig_diff[t, b, :]/dt - sig_diff/dt)**2, axis=2)
      den = np.sum((signal_in[t, b, :] - signal_in)**2, axis=2) + const
      Q[t, b] = np.max(num/den)
  return Q

def percent_tangling(x_in, y_in, th=1):
  """
    scalar summary of tangling() results
    percent tangling, i.e. percentage of points above the y=x line
    Inputs
      x_in: (T, batch_size, n) signal
      y_in: (T, batch_size, n) reference signal
      th: threshold
    Outputs
      percent tangling scalar
  """
  q_emg = tangling(y_in)
  q_m1 = tangling(x_in)
  ratio = q_m1/q_emg
  return np.true_divide(np.sum(ratio > th), ratio.size)

def mean_tangling(x_in, alpha=0.1):
  """
    scalar summary of tangling() results
    mean tangling, i.e. just the mean...
    Inputs
      x_in: (T, batch_size, n) signal
      alpha: fed to tangling()
  """
  return np.mean(tangling(x_in, alpha=alpha))

def dist_tangling(x_in, y_in, alpha=0.1):
  """
    scalar summary of tangling() results
    sum of signed squared distances of [x_in, y_in] from diagonal
    todo: sum of signed squared distances?
  """
  q_y = tangling(y_in, alpha=alpha)
  q_x = tangling(x_in, alpha=alpha)
  q = np.array((q_y.flatten(), q_x.flatten()))
  ones = np.ones_like(q)
  sign = (q[0,:] > q[1,:]).astype('int')
  sign[sign==0] = -1 # above or below diagonal?
  # next line just implements distance calculation. maybe not efficient.
  return np.sum(sign*np.linalg.norm(np.diag(np.dot(q.T, ones))/2.*ones - q, axis=0))
  
def mean_curvature_osculating(signal_in, filt_freq=None):
  """
    mean curvature based on get_curvature() function
  """
  if filt_freq is not None:
    from scipy import signal
    b, a = signal.butter(4, filt_freq)
    signal_in = signal.filtfilt(b, a, signal_in, axis=0)

  C = signal_in.shape[1]
  k = np.zeros(signal_in.shape[:-1])
  for c in range(C):
    k[:, c] = get_curvature(signal_in[:, c, :])
  k[k==0] = np.nan
  return np.nanmean(k, axis=(0,1))
  
def normalize_path(signal_in, filt_freq=0.3):
  """
    normalizes path to length 1
    normalizes across all conditions/batches -- might want to change later
  """
  return signal_in/np.sum(get_path_length(signal_in, filt_freq=filt_freq))
  
def mean_curvature(signal_in, total_points=11, deg=4, normalize=False):
  """
    mean curvature based on get_generalized_curvature() function
  """
  if normalize:
    signal_in = normalize_path(signal_in)
  C = signal_in.shape[1]
  k = np.zeros(signal_in.shape[:-1]+(deg-1,))
  for c in range(C):
    k[:, c, :], _ = get_generalized_curvature(signal_in[:, c, :], total_points, deg)
  k[k==0] = np.nan
  return np.nanmean(k, axis=(0,1))

def get_participation_ratio(signal_in):
  """
    Participation ratio is an approximate measure of dimensionality.
    PR = sum(evs)**2/sum(evs**2)
    where evs are the eigenvalues of the covariance matrix of signal_in
    signal_in is shape (T, batch_size, n), which is reshaped to (T*batch_size, n)
    the covariance matrix is size (n, n)
  """
  signal_in = np.reshape(signal_in, [-1, signal_in.shape[-1]], order='F')
  Cov = np.cov(signal_in.T)
  eigvals = np.linalg.eigvals(Cov)
  return np.sum(eigvals)**2/np.sum(eigvals**2)

def tanh(x):
  return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
def dtanh(x):
  ''' derivative of tanh() '''
  return 1. - tanh(x)**2

def get_jacobian(sim):
  """
    get the jacobian matrix of the RNN
    output: J, a function that takes x, u as vector inputs and returns the jacobian evaluated at those points
  """
  # get A, B
  TF_PATH = RUN+'tfsaves/'+str(sim)
  tf.reset_default_graph()
  with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(TF_PATH+'.meta')
    new_saver.restore(sess, TF_PATH)
    # Get Mat variable
    Mat = sess.run([v for v in tf.all_variables() if v.name == 'RNN/BasicRNNCellNoise/Linear/Matrix:0'][0])
    A = Mat[2:]
    B = Mat[:2]
  def J(x, u):
    return np.dot(A, np.diag(dtanh(np.dot(A.T, x) + np.dot(B.T, u))))
  return J

def get_sum_of_jacobians(sim, x_in, u_in, norm='fro', squared=True):
  """
    get the jacobian and evaluate it at every point in x_in (T, batch_size, n).
    take the norm of the jacobian using `norm` (use 'trace' to caculate divergence)
    average across T and batch_size.
  """
  J = get_jacobian(sim)
  R = 0
  for t in range(x_in.shape[0]):
    for c in range(x_in.shape[1]):
      J_ = J(x_in[t, c, :], u_in[t, c, :])
      if norm == 'trace':
        R_ = np.trace(J_)
      else:
        R_ = np.linalg.norm(J_, norm)
      if squared:
        R_ = R_**2 
      R += R_
  R = np.true_divide(R, np.prod(x_in.shape[:2]))
  return R

