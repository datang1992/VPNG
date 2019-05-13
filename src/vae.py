import itertools
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import slim
import time
from scipy.sparse.linalg import eigsh

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from tensorflow.python.client import timeline


sg = tf.contrib.bayesflow.stochastic_graph
st = tf.contrib.bayesflow.stochastic_tensor
distributions = tf.distributions


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/dat/', 'Directory for data')
flags.DEFINE_string('logdir', '/tmp/log/', 'Directory for logs')

flags.DEFINE_integer('latent_dim', 100, 'Latent dimensionality of model')
flags.DEFINE_integer('batch_size', 600, 'Minibatch size')
flags.DEFINE_integer('validation_size', 0, 'Size of the validation set')
flags.DEFINE_integer('train_subset_size', 10000, 'Size of the subset of the training set')
flags.DEFINE_integer('n_samples', 1, 'Number of samples to save')
flags.DEFINE_integer('n_samples_fisher', 10, 'Number of samples for fisher info')
flags.DEFINE_integer('mc_samples', 10, 'Number of samples to compute expectations using Monte Carlo method')
flags.DEFINE_integer('print_every', 100, 'Print every n iterations')
flags.DEFINE_integer('hidden_size', 200, 'Hidden size for neural networks')
flags.DEFINE_integer('n_iterations_r_fisher', 1500, 'number of iterations for r fisher info')
flags.DEFINE_integer('n_iterations_q_fisher', 2000, 'number of iterations for q fisher info')
flags.DEFINE_integer('n_iterations_normal', 3000, 'number of iterations for normal SGD')
flags.DEFINE_integer('strides', 1, 'Pooling strides')
flags.DEFINE_integer('figure_size', (27 + flags.FLAGS.strides) / flags.FLAGS.strides, 'Number of pixels in a row')
flags.DEFINE_float('reg_ratio', 1e-1, 'Regularization Coefficient')
flags.DEFINE_bool('profile', False, 'Do profiling on the training process or not')
flags.DEFINE_float('learning_rate_r_fisher', 0.01, 'Learning rate for r fisher info')
flags.DEFINE_float('learning_rate_q_fisher', 0.002, 'Learning rate for q fisher info')
flags.DEFINE_float('learning_rate_normal', 0.003, 'Learning rate for normal SGD')
flags.DEFINE_string('output_file_name_r_fisher', None, 'The output file path for fisher info')
flags.DEFINE_string('output_file_name_q_fisher', None, 'The output file path for fisher info')
flags.DEFINE_string('output_file_name_normal', None, 'The output file path for normal SGD')
flags.DEFINE_float('num_eigs_ln_ratio', 4., 'Ratio for number of eigenvectors, with ln computation')
flags.DEFINE_float('ema_decay', .9, 'Decay of exponential moving average')
flags.DEFINE_float('Epsilon', 1e-6, 'Value for big Epsilon')
flags.DEFINE_float('epsilon', 1e-8, 'Value for small epsilon')
flags.DEFINE_float('init_scale', 0.1, 'Normal std value for variable initialization')

FLAGS = flags.FLAGS

def my_eigsh_v2(A):
  n = np.shape(A)[0]
  num_eigs = int(FLAGS.num_eigs_ln_ratio * np.log(n))
  return eigsh(A, num_eigs)

def kronecker_prod_sum_matrix_solve_v3(A_sqrt_i, B_sqrt_i, C, D, V, M1_eigsh=False):
  # Solving the equation (A\oprod B + C\oprod D)x = vec(V).
  # Assume that A, B, C, D are positive semi-definite. Also, A, B are invertible.
  # A, B, C, D should be 2D tensors. V should be a tensor of size n * m,
  # where A, C are m * m and B, D are n * n matrices.
  M1 = tf.matmul(A_sqrt_i, tf.matmul(C, A_sqrt_i))
  M2 = tf.matmul(B_sqrt_i, tf.matmul(D, B_sqrt_i))
  if M1_eigsh:
    S1, E1 = tf.py_func(my_eigsh_v2, [(M1 + tf.transpose(M1)) / 2.0], [tf.float32] * 2)
  else:
    S1, E1 = tf.py_func(np.linalg.eigh, [(M1 + tf.transpose(M1)) / 2.0], [tf.float32] * 2)
  S2, E2 = tf.py_func(np.linalg.eigh, [(M2 + tf.transpose(M2)) / 2.0], [tf.float32] * 2)
  K1 = tf.matmul(A_sqrt_i, E1)
  K2 = tf.matmul(B_sqrt_i, E2)
  return  tf.matmul(K2, tf.matmul(tf.matmul(tf.transpose(K2), tf.matmul(V, K1)) / (1.0
            - tf.reshape(S2, [-1, 1]) * tf.reshape(S1, [1, -1])), tf.transpose(K1)))

def weight_variable(shape, name, reuse=False, init_val_dict=None):
  if reuse:
    with tf.variable_scope("weights", reuse=True):
      variable = tf.get_variable(name)
  else:
    if name in init_val_dict:
      init_val = init_val_dict[name]
    else:
      init_val = tf.truncated_normal(shape, stddev=FLAGS.init_scale)
      init_val_dict[name] = init_val
    with tf.variable_scope("weights", reuse=False):
      variable = tf.get_variable(name, initializer=init_val)
  return variable

def inference_network(x, latent_dim, hidden_size, figure_size, init_val_dict=None):
  """Construct an inference network parametrizing a Gaussian.
  Args:
    x: A batch of MNIST digits.
    latent_dim: The latent dimensionality.
    hidden_size: The size of the neural net hidden layers.
  Returns:
    mu: Mean parameters for the variational family Normal
    sigma: Standard deviation parameters for the variational family Normal
  """
  h0 = slim.flatten(x)
  h0b = tf.concat([h0, tf.ones([tf.shape(h0)[0], 1])], 1)
  W0b = weight_variable([figure_size * figure_size + 1, hidden_size], "W0bi", init_val_dict=init_val_dict)
  h1 = tf.nn.relu(tf.matmul(h0b, W0b))
  h1b = tf.concat([h1, tf.ones([tf.shape(h1)[0], 1])], 1)
  W1b = weight_variable([hidden_size + 1, hidden_size], "W1bi", init_val_dict=init_val_dict)
  h2 = tf.nn.relu(tf.matmul(h1b, W1b))
  h2b = tf.concat([h2, tf.ones([tf.shape(h2)[0], 1])], 1)
  W2b = weight_variable([hidden_size + 1, latent_dim * 2], "W2bi", init_val_dict=init_val_dict)
  h3 = tf.matmul(h2b, W2b)
  q_mu = h3[:, :latent_dim]
  q_sigma_raw = h3[:, latent_dim:]
  return q_mu, q_sigma_raw, h2b, h2, h1b, h1, h0b, W2b, W1b, W0b


def generative_network(z, hidden_size, latent_dim, figure_size, reuse=False, need_reshape=False, init_val_dict=None):
  """Build a generative network parametrizing the likelihood of the data
  Args:
    z: Samples of latent variables
    hidden_size: Size of the hidden state of the neural net
  Returns:
    bernoulli_logits: logits for the Bernoulli likelihood of the data
  """
  W0b = weight_variable([latent_dim + 1, hidden_size], "W0bg", reuse=reuse, init_val_dict=init_val_dict)
  if need_reshape:
    h0b = tf.reshape(tf.concat([z, tf.ones([tf.shape(z)[0], tf.shape(z)[1], 1])], 2), [-1, latent_dim + 1])
  else:
    h0b = tf.concat([z, tf.ones([tf.shape(z)[0], 1])], 1)
  h1 = tf.nn.relu(tf.matmul(h0b, W0b))
  h1b = tf.concat([h1, tf.ones([tf.shape(h1)[0], 1])], 1)
  W1b = weight_variable([hidden_size + 1, hidden_size], "W1bg", reuse=reuse, init_val_dict=init_val_dict)
  h2 = tf.nn.relu(tf.matmul(h1b, W1b))
  h2b = tf.concat([h2, tf.ones([tf.shape(h2)[0], 1])], 1)
  W2b = weight_variable([hidden_size + 1, figure_size * figure_size], "W2bg", reuse=reuse, init_val_dict=init_val_dict)
  h3 = tf.matmul(h2b, W2b)
  if need_reshape:
    bernoulli_logits = tf.reshape(h3, [-1, tf.shape(z)[1], figure_size, figure_size, 1])
  else:
    bernoulli_logits = tf.reshape(h3, [-1, figure_size, figure_size, 1])
  return bernoulli_logits, h3, h2b, h2, h1b, h1, h0b, W2b, W1b, W0b

def train():
  with tf.name_scope('data'):
    x = tf.placeholder(tf.float32, [None, FLAGS.figure_size, FLAGS.figure_size, 1])
    tf.summary.image('data', x)

  iter_num = tf.placeholder(tf.int32, [])

  para_size = (FLAGS.figure_size * FLAGS.figure_size + 1) * FLAGS.hidden_size \
                        + (FLAGS.hidden_size + 1) * FLAGS.hidden_size \
                        + (FLAGS.hidden_size + 1) * FLAGS.latent_dim * 2 \
                        + (FLAGS.latent_dim + 1) * FLAGS.hidden_size \
                        + (FLAGS.hidden_size + 1) * FLAGS.hidden_size \
                        + (FLAGS.hidden_size + 1) * FLAGS.figure_size * FLAGS.figure_size

  print 'Total paramter size: ' + str(para_size)

  init_val_dict = {}

  with tf.variable_scope('variational_normal'):
    q_mu_, q_sigma_raw_, _, _, _, _, _, _, _, _ = inference_network(x=x,
                              latent_dim=FLAGS.latent_dim,
                              hidden_size=FLAGS.hidden_size,
                              figure_size=FLAGS.figure_size,
                              init_val_dict=init_val_dict)
    q_sigma_ = FLAGS.Epsilon + tf.nn.softplus(q_sigma_raw_)
    with st.value_type(st.SampleValue(FLAGS.mc_samples)):
      q_z_ = st.StochasticTensor(distributions.Normal(loc=q_mu_, scale=q_sigma_))

  with tf.variable_scope('variational_q_fisher'):
    q_mu__, q_sigma_raw__, h2bi__, h2i__, h1bi__, h1i__, h0bi__, W2bi__, W1bi__, W0bi__ = inference_network(x=x,
                              latent_dim=FLAGS.latent_dim,
                              hidden_size=FLAGS.hidden_size,
                              figure_size=FLAGS.figure_size,
                              init_val_dict=init_val_dict)
    q_sigma__ = FLAGS.Epsilon + tf.nn.softplus(q_sigma_raw__)
    with st.value_type(st.SampleValue(FLAGS.mc_samples)):
      q_z__ = st.StochasticTensor(distributions.Normal(loc=q_mu__, scale=q_sigma__))
    with st.value_type(st.SampleValue(FLAGS.n_samples_fisher)):
      q_z3__ = st.StochasticTensor(distributions.Normal(loc=q_mu__, scale=q_sigma__))

  with tf.variable_scope('variational_r_fisher'):
    q_mu, q_sigma_raw, h2bi, h2i, h1bi, h1i, h0bi, W2bi, W1bi, W0bi = inference_network(x=x,
                              latent_dim=FLAGS.latent_dim,
                              hidden_size=FLAGS.hidden_size,
                              figure_size=FLAGS.figure_size,
                              init_val_dict=init_val_dict)
    q_sigma = FLAGS.Epsilon + tf.nn.softplus(q_sigma_raw)
    with st.value_type(st.SampleValue(FLAGS.mc_samples)):
      q_z = st.StochasticTensor(distributions.Normal(loc=q_mu, scale=q_sigma))

  with tf.variable_scope('model_normal'):
    p_x_given_z_logits_ = generative_network(z=q_z_,
                                            hidden_size=FLAGS.hidden_size,
                                            latent_dim=FLAGS.latent_dim,
                                            figure_size=FLAGS.figure_size,
                                            reuse=False,
                                            need_reshape=True,
                                            init_val_dict=init_val_dict)[0]
    p_x_given_z_ = distributions.Bernoulli(logits=p_x_given_z_logits_)

  with tf.variable_scope('model_q_fisher'):
    p_x_given_z_logits__, _, _, _, _, _, _, W2bg__, W1bg__, W0bg__  = generative_network(z=q_z__,
                                            hidden_size=FLAGS.hidden_size,
                                            latent_dim=FLAGS.latent_dim,
                                            figure_size=FLAGS.figure_size,
                                            reuse=False,
                                            need_reshape=True,
                                            init_val_dict=init_val_dict)
    p_x_given_z__ = distributions.Bernoulli(logits=p_x_given_z_logits__)

  with tf.variable_scope('model_r_fisher'):
    p_x_given_z_logits = generative_network(z=q_z,
                                            hidden_size=FLAGS.hidden_size,
                                            latent_dim=FLAGS.latent_dim,
                                            figure_size=FLAGS.figure_size,
                                            reuse=False,
                                            need_reshape=True,
                                            init_val_dict=init_val_dict)[0]
    p_x_given_z = distributions.Bernoulli(logits=p_x_given_z_logits)

  p_z = distributions.Normal(loc=np.zeros(FLAGS.latent_dim, dtype=np.float32),
                              scale=np.ones(FLAGS.latent_dim, dtype=np.float32))

  kl_ = tf.reduce_sum(distributions.kl_divergence(q_z_.distribution, p_z), 1)
  expected_log_likelihood_ = tf.reduce_sum(p_x_given_z_.log_prob(x),
                                          [0, 2, 3, 4]) / float(FLAGS.mc_samples)
  elbo_ = tf.reduce_mean(expected_log_likelihood_ - kl_)
  optimizer_ = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_normal, epsilon=FLAGS.epsilon)
  
  kl__ = tf.reduce_sum(distributions.kl_divergence(q_z__.distribution, p_z), 1)
  expected_log_likelihood__ = tf.reduce_sum(p_x_given_z__.log_prob(x),
                                          [0, 2, 3, 4]) / float(FLAGS.mc_samples)
  elbo__ = tf.reduce_mean(expected_log_likelihood__ - kl__)
  optimizer__ = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_q_fisher, epsilon=FLAGS.epsilon)
  
  kl = tf.reduce_sum(distributions.kl_divergence(q_z.distribution, p_z), 1)
  expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x),
                                          [0, 2, 3, 4]) / float(FLAGS.mc_samples)
  elbo = tf.reduce_mean(expected_log_likelihood - kl)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_r_fisher, epsilon=FLAGS.epsilon)

  with st.value_type(st.SampleValue()):
    q_z3 = st.StochasticTensor(distributions.Normal(loc=q_mu, scale=q_sigma))
    with tf.variable_scope('model_r_fisher', reuse=True):
      p_x_given_z_logits3, h3g, h2bg, h2g, h1bg, h1g, h0bg, W2bg, W1bg, W0bg = generative_network(z=q_z3,
                                               hidden_size=FLAGS.hidden_size,
                                               latent_dim=FLAGS.latent_dim,
                                               figure_size=FLAGS.figure_size,
                                               reuse=True)
  train_op_ = optimizer_.minimize(-elbo_)
  
  (grad_W0bi__, grad_W1bi__, grad_W2bi__, grad_W0bg__, grad_W1bg__, grad_W2bg__) = tf.gradients(-elbo__, [W0bi__, W1bi__, W2bi__, W0bg__, W1bg__, W2bg__])
  h0bi_t__ = tf.transpose(h0bi__)
  h1bi_t__ = tf.transpose(h1bi__)
  h2bi_t__ = tf.transpose(h2bi__)
  ema__ = tf.train.ExponentialMovingAverage(decay=FLAGS.ema_decay, num_updates=iter_num)
  with tf.variable_scope('average_q'):  
    A00b__ = tf.get_variable('A00b', shape=[FLAGS.figure_size * FLAGS.figure_size + 1, FLAGS.figure_size * FLAGS.figure_size + 1], initializer=tf.zeros_initializer())
    A01b__ = tf.get_variable('A01b', shape=[FLAGS.figure_size * FLAGS.figure_size + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    A11b__ = tf.get_variable('A11b', shape=[FLAGS.hidden_size + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    A12b__ = tf.get_variable('A12b', shape=[FLAGS.hidden_size + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    A22b__ = tf.get_variable('A22b', shape=[FLAGS.hidden_size + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    G11__ = tf.get_variable('G11', shape=[FLAGS.hidden_size, FLAGS.hidden_size], initializer=tf.zeros_initializer())
    G12__ = tf.get_variable('G12', shape=[FLAGS.hidden_size, FLAGS.hidden_size], initializer=tf.zeros_initializer())
    G22__ = tf.get_variable('G22', shape=[FLAGS.hidden_size, FLAGS.hidden_size], initializer=tf.zeros_initializer())
    G23__ = tf.get_variable('G23', shape=[FLAGS.hidden_size, FLAGS.latent_dim * 2], initializer=tf.zeros_initializer())
    G33_diag__ = tf.get_variable('G33_diag', shape=[FLAGS.latent_dim * 2], initializer=tf.zeros_initializer())
  ema_apply_op__ = ema__.apply([A00b__, A01b__, A11b__, A12b__, A22b__, G11__, G12__, G22__, G23__, G33_diag__])
  A00b_v__ = ema__.average(A00b__)
  A01b_v__ = ema__.average(A01b__)
  A11b_v__ = ema__.average(A11b__)
  A12b_v__ = ema__.average(A12b__)
  A22b_v__ = ema__.average(A22b__)
  G11_v__ = ema__.average(G11__)
  G12_v__ = ema__.average(G12__)
  G22_v__ = ema__.average(G22__)
  G23_v__ = ema__.average(G23__)
  G33_diag_v__ = ema__.average(G33_diag__)
  A00b_new__ = tf.matmul(h0bi_t__, h0bi__) / FLAGS.batch_size
  A01b_new__ = tf.matmul(h0bi_t__, h1bi__) / FLAGS.batch_size
  A11b_new__ = tf.matmul(h1bi_t__, h1bi__) / FLAGS.batch_size
  A12b_new__ = tf.matmul(h1bi_t__, h2bi__) / FLAGS.batch_size
  A22b_new__ = tf.matmul(h2bi_t__, h2bi__) / FLAGS.batch_size
  S_A00b__, E_A00b__ = tf.py_func(my_eigsh_v2, [(A00b_v__ + tf.transpose(A00b_v__)) / 2.0], [tf.float32] * 2)
  S_A00b_i__ = 1.0 / (S_A00b__ + FLAGS.reg_ratio)
  E_A00b_T__ = tf.transpose(E_A00b__)
  A00b_i__ = tf.matmul(E_A00b__ * S_A00b_i__, E_A00b_T__)
  A00b_sqrt_i__ = tf.matmul(E_A00b__ * tf.sqrt(S_A00b_i__), E_A00b_T__)
  S_A11b__, E_A11b__ = tf.py_func(np.linalg.eigh, [(A11b_v__ + tf.transpose(A11b_v__)) / 2.0], [tf.float32] * 2)
  S_A11b_i__ = 1.0 / (S_A11b__ + FLAGS.reg_ratio)
  E_A11b_T__ = tf.transpose(E_A11b__)
  A11b_i__ = tf.matmul(E_A11b__ * S_A11b_i__, E_A11b_T__)
  A11b_sqrt_i__ = tf.matmul(E_A11b__ * tf.sqrt(S_A11b_i__), E_A11b_T__)
  S_A22b__, E_A22b__ = tf.py_func(np.linalg.eigh, [(A22b_v__ + tf.transpose(A22b_v__)) / 2.0], [tf.float32] * 2)
  S_A22b_i__ = 1.0 / (S_A22b__ + FLAGS.reg_ratio)
  E_A22b_T__ = tf.transpose(E_A22b__)
  A22b_i__ = tf.matmul(E_A22b__ * S_A22b_i__, E_A22b_T__)
  A22b_sqrt_i__ = tf.matmul(E_A22b__ * tf.sqrt(S_A22b_i__), E_A22b_T__)
  grad_mat_original3__ = tf.concat([(q_mu__ - q_z3__) / (q_sigma__ ** 2),\
                          (1.0 / q_sigma__ - ((q_mu__ - q_z3__) ** 2) / (q_sigma__ ** 3)) * tf.nn.sigmoid(q_sigma_raw__)], 2)
  grad_mat_original2__ = tf.reshape(tf.matmul(tf.reshape(grad_mat_original3__, [-1, FLAGS.latent_dim * 2]), tf.transpose(W2bi__[:-1])),
                          [-1, FLAGS.batch_size, FLAGS.hidden_size]) * tf.cast(h2i__ > 0, tf.float32)
  grad_mat_original1__ = tf.reshape(tf.matmul(tf.reshape(grad_mat_original2__, [-1, FLAGS.hidden_size]), tf.transpose(W1bi__[:-1])),
                          [-1, FLAGS.batch_size, FLAGS.hidden_size]) * tf.cast(h1i__ > 0, tf.float32)
  grad_mat_original3_f__ = tf.reshape(grad_mat_original3__, [-1, FLAGS.latent_dim * 2])
  grad_mat_original2_f__ = tf.reshape(grad_mat_original2__, [-1, FLAGS.hidden_size])
  grad_mat_original1_f__ = tf.reshape(grad_mat_original1__, [-1, FLAGS.hidden_size])
  grad_mat_original3_ft__ = tf.transpose(grad_mat_original3_f__)
  grad_mat_original2_ft__ = tf.transpose(grad_mat_original2_f__)
  grad_mat_original1_ft__ = tf.transpose(grad_mat_original1_f__)
  G11_new__ = tf.matmul(grad_mat_original1_ft__, grad_mat_original1_f__) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G12_new__ = tf.matmul(grad_mat_original1_ft__, grad_mat_original2_f__) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G22_new__ = tf.matmul(grad_mat_original2_ft__, grad_mat_original2_f__) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G23_new__ = tf.matmul(grad_mat_original2_ft__, grad_mat_original3_f__) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G33_diag_new__ = tf.reduce_mean(tf.concat([1.0 / (q_sigma__ ** 2), 2.0 * ((tf.nn.sigmoid(q_sigma_raw__) / q_sigma__) ** 2)], 1), 0)
  S_G11__, E_G11__ = tf.py_func(np.linalg.eigh, [(G11_v__ + tf.transpose(G11_v__)) / 2.0], [tf.float32] * 2)
  S_G11_i__ = 1.0 / (S_G11__ + FLAGS.reg_ratio)
  E_G11_T__ = tf.transpose(E_G11__)
  G11_i__ = tf.matmul(E_G11__ * S_G11_i__, E_G11_T__)
  G11_sqrt_i__ = tf.matmul(E_G11__ * tf.sqrt(S_G11_i__), E_G11_T__)
  S_G22__, E_G22__ = tf.py_func(np.linalg.eigh, [(G22_v__ + tf.transpose(G22_v__)) / 2.0], [tf.float32] * 2)
  S_G22_i__ = 1.0 / (S_G22__ + FLAGS.reg_ratio)
  E_G22_T__ = tf.transpose(E_G22__)
  G22_i__ = tf.matmul(E_G22__ * S_G22_i__, E_G22_T__)
  G22_sqrt_i__ = tf.matmul(E_G22__ * tf.sqrt(S_G22_i__), E_G22_T__)
  G33_i_diag__ = 1.0 / (G33_diag_v__ + FLAGS.reg_ratio)
  psi_Ab_01__ = tf.matmul(A01b_v__, A11b_i__)
  psi_Ab_12__ = tf.matmul(A12b_v__, A22b_i__)
  psi_G_12__ = tf.matmul(G12_v__, G22_i__)
  psi_G_23__ = G23_v__ * G33_i_diag__
  V1__ = tf.transpose(grad_W0bi__)
  V2__ = tf.transpose(grad_W1bi__)
  V3__ = tf.transpose(grad_W2bi__)
  V1__ = V1__ - tf.matmul(psi_G_12__, tf.matmul(V2__, tf.transpose(psi_Ab_01__)))
  V2__ = V2__ - tf.matmul(psi_G_23__, tf.matmul(V3__, tf.transpose(psi_Ab_12__)))
  C01__ = tf.matmul(psi_Ab_01__, tf.transpose(A01b_v__))
  C12__ = tf.matmul(psi_Ab_12__, tf.transpose(A12b_v__))
  D12__ = tf.matmul(psi_G_12__, tf.transpose(G12_v__))
  D23__ = tf.matmul(psi_G_23__, tf.transpose(G23_v__))
  V1__ = kronecker_prod_sum_matrix_solve_v3(A00b_sqrt_i__, G11_sqrt_i__, C01__, D12__, V1__, True)
  V2__ = kronecker_prod_sum_matrix_solve_v3(A11b_sqrt_i__, G22_sqrt_i__, C12__, D23__, V2__)
  V3__ = tf.reshape(G33_i_diag__, [-1, 1]) * tf.matmul(V3__, A22b_i__)
  V3__ = V3__ - tf.matmul(tf.transpose(psi_G_23__), tf.matmul(V2__, psi_Ab_12__))
  V2__ = V2__ - tf.matmul(tf.transpose(psi_G_12__), tf.matmul(V1__, psi_Ab_01__))
  ema_assign_op__ = [tf.assign(A00b__, A00b_new__), tf.assign(A01b__, A01b_new__), tf.assign(A11b__, A11b_new__), tf.assign(A12b__, A12b_new__),\
                tf.assign(A22b__, A22b_new__), tf.assign(G11__, G11_new__), tf.assign(G12__, G12_new__), tf.assign(G22__, G22_new__),\
                tf.assign(G23__, G23_new__), tf.assign(G33_diag__, G33_diag_new__)]
  train_op__ = optimizer__.apply_gradients([(tf.transpose(V1__), W0bi__), (tf.transpose(V2__), W1bi__), (tf.transpose(V3__), W2bi__),\
                                            (grad_W0bg__, W0bg__), (grad_W1bg__, W1bg__), (grad_W2bg__, W2bg__)])


  (grad_W0bi, grad_W1bi, grad_W2bi, grad_W0bg, grad_W1bg, grad_W2bg) = tf.gradients(-elbo, [W0bi, W1bi, W2bi, W0bg, W1bg, W2bg])
  p_x_given_z_logits3v = tf.reshape(p_x_given_z_logits3, [FLAGS.batch_size, -1])
  p_x_given_z3 = distributions.Bernoulli(logits=p_x_given_z_logits3v)
  p_x_given_z3_samples = p_x_given_z3.sample(FLAGS.n_samples_fisher)
  p_x_given_z3v_1 = tf.nn.sigmoid(p_x_given_z_logits3v)
  p_x_given_z3v_prob = (1.0 - p_x_given_z3v_1) * p_x_given_z3v_1
  h0bi_t = tf.transpose(h0bi)
  h1bi_t = tf.transpose(h1bi)
  h2bi_t = tf.transpose(h2bi)
  h0bg_t = tf.transpose(h0bg)
  h1bg_t = tf.transpose(h1bg)
  h2bg_t = tf.transpose(h2bg)
  ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ema_decay, num_updates=iter_num)
  with tf.variable_scope('average_r'):  
    A00b = tf.get_variable('A00b', shape=[FLAGS.figure_size * FLAGS.figure_size + 1, FLAGS.figure_size * FLAGS.figure_size + 1], initializer=tf.zeros_initializer())
    A01b = tf.get_variable('A01b', shape=[FLAGS.figure_size * FLAGS.figure_size + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    A11b = tf.get_variable('A11b', shape=[FLAGS.hidden_size + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    A12b = tf.get_variable('A12b', shape=[FLAGS.hidden_size + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    A22b = tf.get_variable('A22b', shape=[FLAGS.hidden_size + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    A23b = tf.get_variable('A23b', shape=[FLAGS.hidden_size + 1, FLAGS.latent_dim + 1], initializer=tf.zeros_initializer())
    A33b = tf.get_variable('A33b', shape=[FLAGS.latent_dim + 1, FLAGS.latent_dim + 1], initializer=tf.zeros_initializer())
    A34b = tf.get_variable('A34b', shape=[FLAGS.latent_dim + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    A44b = tf.get_variable('A44b', shape=[FLAGS.hidden_size + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    A45b = tf.get_variable('A45b', shape=[FLAGS.hidden_size + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    A55b = tf.get_variable('A55b', shape=[FLAGS.hidden_size + 1, FLAGS.hidden_size + 1], initializer=tf.zeros_initializer())
    G11 = tf.get_variable('G11', shape=[FLAGS.hidden_size, FLAGS.hidden_size], initializer=tf.zeros_initializer())
    G12 = tf.get_variable('G12', shape=[FLAGS.hidden_size, FLAGS.hidden_size], initializer=tf.zeros_initializer())
    G22 = tf.get_variable('G22', shape=[FLAGS.hidden_size, FLAGS.hidden_size], initializer=tf.zeros_initializer())
    G23 = tf.get_variable('G23', shape=[FLAGS.hidden_size, FLAGS.latent_dim * 2], initializer=tf.zeros_initializer())
    G33 = tf.get_variable('G33', shape=[FLAGS.latent_dim * 2, FLAGS.latent_dim * 2], initializer=tf.zeros_initializer())
    G34 = tf.get_variable('G34', shape=[FLAGS.latent_dim * 2, FLAGS.hidden_size], initializer=tf.zeros_initializer())
    G44 = tf.get_variable('G44', shape=[FLAGS.hidden_size, FLAGS.hidden_size], initializer=tf.zeros_initializer())
    G45 = tf.get_variable('G45', shape=[FLAGS.hidden_size, FLAGS.hidden_size], initializer=tf.zeros_initializer())
    G55 = tf.get_variable('G55', shape=[FLAGS.hidden_size, FLAGS.hidden_size], initializer=tf.zeros_initializer())
    G56 = tf.get_variable('G56', shape=[FLAGS.hidden_size, FLAGS.figure_size * FLAGS.figure_size], initializer=tf.zeros_initializer())
    G66_diag = tf.get_variable('G66_diag', shape=[FLAGS.figure_size * FLAGS.figure_size], initializer=tf.zeros_initializer())
  ema_apply_op = ema.apply([A00b, A01b, A11b, A12b, A22b, A23b, A33b, A34b, A44b, A45b, A55b, G11, G12, G22, G23, G33, G34, G44, G45, G55, G56, G66_diag])
  A00b_v = ema.average(A00b)
  A01b_v = ema.average(A01b)
  A11b_v = ema.average(A11b)
  A12b_v = ema.average(A12b)
  A22b_v = ema.average(A22b)
  A23b_v = ema.average(A23b)
  A33b_v = ema.average(A33b)
  A34b_v = ema.average(A34b)
  A44b_v = ema.average(A44b)
  A45b_v = ema.average(A45b)
  A55b_v = ema.average(A55b)
  G11_v = ema.average(G11)
  G12_v = ema.average(G12)
  G22_v = ema.average(G22)
  G23_v = ema.average(G23)
  G33_v = ema.average(G33)
  G34_v = ema.average(G34)
  G44_v = ema.average(G44)
  G45_v = ema.average(G45)
  G55_v = ema.average(G55)
  G56_v = ema.average(G56)
  G66_diag_v = ema.average(G66_diag)

  A00b_new = tf.matmul(h0bi_t, h0bi) / FLAGS.batch_size
  A01b_new = tf.matmul(h0bi_t, h1bi) / FLAGS.batch_size
  A11b_new = tf.matmul(h1bi_t, h1bi) / FLAGS.batch_size
  A12b_new = tf.matmul(h1bi_t, h2bi) / FLAGS.batch_size
  A22b_new = tf.matmul(h2bi_t, h2bi) / FLAGS.batch_size
  A23b_new = tf.matmul(h2bi_t, h0bg) / FLAGS.batch_size
  A33b_new = tf.matmul(h0bg_t, h0bg) / FLAGS.batch_size
  A34b_new = tf.matmul(h0bg_t, h1bg) / FLAGS.batch_size
  A44b_new = tf.matmul(h1bg_t, h1bg) / FLAGS.batch_size
  A45b_new = tf.matmul(h1bg_t, h2bg) / FLAGS.batch_size
  A55b_new = tf.matmul(h2bg_t, h2bg) / FLAGS.batch_size

  S_A00b, E_A00b = tf.py_func(my_eigsh_v2, [(A00b_v + tf.transpose(A00b_v)) / 2.0], [tf.float32] * 2)
  S_A00b_i = 1.0 / (S_A00b + FLAGS.reg_ratio)
  E_A00b_T = tf.transpose(E_A00b)
  A00b_i = tf.matmul(E_A00b * S_A00b_i, E_A00b_T)
  A00b_sqrt_i = tf.matmul(E_A00b * tf.sqrt(S_A00b_i), E_A00b_T)
  S_A11b, E_A11b = tf.py_func(np.linalg.eigh, [(A11b_v + tf.transpose(A11b_v)) / 2.0], [tf.float32] * 2)
  S_A11b_i = 1.0 / (S_A11b + FLAGS.reg_ratio)
  E_A11b_T = tf.transpose(E_A11b)
  A11b_i = tf.matmul(E_A11b * S_A11b_i, E_A11b_T)
  A11b_sqrt_i = tf.matmul(E_A11b * tf.sqrt(S_A11b_i), E_A11b_T)
  S_A22b, E_A22b = tf.py_func(np.linalg.eigh, [(A22b_v + tf.transpose(A22b_v)) / 2.0], [tf.float32] * 2)
  S_A22b_i = 1.0 / (S_A22b + FLAGS.reg_ratio)
  E_A22b_T = tf.transpose(E_A22b)
  A22b_i = tf.matmul(E_A22b * S_A22b_i, E_A22b_T)
  A22b_sqrt_i = tf.matmul(E_A22b * tf.sqrt(S_A22b_i), E_A22b_T)
  S_A33b, E_A33b = tf.py_func(np.linalg.eigh, [(A33b_v + tf.transpose(A33b_v)) / 2.0], [tf.float32] * 2)
  S_A33b_i = 1.0 / (S_A33b + FLAGS.reg_ratio)
  E_A33b_T = tf.transpose(E_A33b)
  A33b_i = tf.matmul(E_A33b * S_A33b_i, E_A33b_T)
  A33b_sqrt_i = tf.matmul(E_A33b * tf.sqrt(S_A33b_i), E_A33b_T)
  S_A44b, E_A44b = tf.py_func(np.linalg.eigh, [(A44b_v + tf.transpose(A44b_v)) / 2.0], [tf.float32] * 2)
  S_A44b_i = 1.0 / (S_A44b + FLAGS.reg_ratio)
  E_A44b_T = tf.transpose(E_A44b)
  A44b_i = tf.matmul(E_A44b * S_A44b_i, E_A44b_T)
  A44b_sqrt_i = tf.matmul(E_A44b * tf.sqrt(S_A44b_i), E_A44b_T)
  S_A55b, E_A55b = tf.py_func(np.linalg.eigh, [(A55b_v + tf.transpose(A55b_v)) / 2.0], [tf.float32] * 2)
  S_A55b_i = 1.0 / (S_A55b + FLAGS.reg_ratio)
  E_A55b_T = tf.transpose(E_A55b)
  A55b_i = tf.matmul(E_A55b * S_A55b_i, E_A55b_T)
  A55b_sqrt_i = tf.matmul(E_A55b * tf.sqrt(S_A55b_i), E_A55b_T)
  grad_mat_original6 = tf.cast(p_x_given_z3_samples, tf.float32) - p_x_given_z3v_1
  grad_mat_original5 = tf.reshape(tf.matmul(tf.reshape(grad_mat_original6, [-1, FLAGS.figure_size * FLAGS.figure_size]), tf.transpose(W2bg[:-1])),
                          [-1, FLAGS.batch_size, FLAGS.hidden_size]) * tf.cast(h2g > 0, tf.float32)
  grad_mat_original4 = tf.reshape(tf.matmul(tf.reshape(grad_mat_original5, [-1, FLAGS.hidden_size]), tf.transpose(W1bg[:-1])),
                          [-1, FLAGS.batch_size, FLAGS.hidden_size]) * tf.cast(h1g > 0, tf.float32)
  grad_mat_original3 = tf.reshape(tf.matmul(tf.reshape(grad_mat_original4, [-1, FLAGS.hidden_size]), tf.transpose(W0bg[:-1])),
                          [-1, FLAGS.batch_size, FLAGS.latent_dim])
  grad_mat_original3 = tf.concat([grad_mat_original3, grad_mat_original3 * ((q_z3 - q_mu) / q_sigma) * tf.nn.sigmoid(q_sigma_raw)], 2)
  grad_mat_original2 = tf.reshape(tf.matmul(tf.reshape(grad_mat_original3, [-1, FLAGS.latent_dim * 2]), tf.transpose(W2bi[:-1])),
                          [-1, FLAGS.batch_size, FLAGS.hidden_size]) * tf.cast(h2i > 0, tf.float32)
  grad_mat_original1 = tf.reshape(tf.matmul(tf.reshape(grad_mat_original2, [-1, FLAGS.hidden_size]), tf.transpose(W1bi[:-1])),
                          [-1, FLAGS.batch_size, FLAGS.hidden_size]) * tf.cast(h1i > 0, tf.float32)
  grad_mat_original6_f = tf.reshape(grad_mat_original6, [-1, FLAGS.figure_size * FLAGS.figure_size])
  grad_mat_original5_f = tf.reshape(grad_mat_original5, [-1, FLAGS.hidden_size])
  grad_mat_original4_f = tf.reshape(grad_mat_original4, [-1, FLAGS.hidden_size])
  grad_mat_original3_f = tf.reshape(grad_mat_original3, [-1, FLAGS.latent_dim * 2])
  grad_mat_original2_f = tf.reshape(grad_mat_original2, [-1, FLAGS.hidden_size])
  grad_mat_original1_f = tf.reshape(grad_mat_original1, [-1, FLAGS.hidden_size])
  grad_mat_original6_ft = tf.transpose(grad_mat_original6_f)
  grad_mat_original5_ft = tf.transpose(grad_mat_original5_f)
  grad_mat_original4_ft = tf.transpose(grad_mat_original4_f)
  grad_mat_original3_ft = tf.transpose(grad_mat_original3_f)
  grad_mat_original2_ft = tf.transpose(grad_mat_original2_f)
  grad_mat_original1_ft = tf.transpose(grad_mat_original1_f)
  G11_new = tf.matmul(grad_mat_original1_ft, grad_mat_original1_f) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G12_new = tf.matmul(grad_mat_original1_ft, grad_mat_original2_f) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G22_new = tf.matmul(grad_mat_original2_ft, grad_mat_original2_f) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G23_new = tf.matmul(grad_mat_original2_ft, grad_mat_original3_f) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G33_new = tf.matmul(grad_mat_original3_ft, grad_mat_original3_f) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G34_new = tf.matmul(grad_mat_original3_ft, grad_mat_original4_f) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G44_new = tf.matmul(grad_mat_original4_ft, grad_mat_original4_f) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G45_new = tf.matmul(grad_mat_original4_ft, grad_mat_original5_f) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G55_new = tf.matmul(grad_mat_original5_ft, grad_mat_original5_f) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G56_new = tf.matmul(grad_mat_original5_ft, grad_mat_original6_f) / (FLAGS.n_samples_fisher * FLAGS.batch_size)
  G66_diag_new = tf.reduce_mean(p_x_given_z3v_prob, 0)

  S_G11, E_G11 = tf.py_func(np.linalg.eigh, [(G11_v + tf.transpose(G11_v)) / 2.0], [tf.float32] * 2)
  S_G11_i = 1.0 / (S_G11 + FLAGS.reg_ratio)
  E_G11_T = tf.transpose(E_G11)
  G11_i = tf.matmul(E_G11 * S_G11_i, E_G11_T)
  G11_sqrt_i = tf.matmul(E_G11 * tf.sqrt(S_G11_i), E_G11_T)
  S_G22, E_G22 = tf.py_func(np.linalg.eigh, [(G22_v + tf.transpose(G22_v)) / 2.0], [tf.float32] * 2)
  S_G22_i = 1.0 / (S_G22 + FLAGS.reg_ratio)
  E_G22_T = tf.transpose(E_G22)
  G22_i = tf.matmul(E_G22 * S_G22_i, E_G22_T)
  G22_sqrt_i = tf.matmul(E_G22 * tf.sqrt(S_G22_i), E_G22_T)
  S_G33, E_G33 = tf.py_func(np.linalg.eigh, [(G33_v + tf.transpose(G33_v)) / 2.0], [tf.float32] * 2)
  S_G33_i = 1.0 / (S_G33 + FLAGS.reg_ratio)
  E_G33_T = tf.transpose(E_G33)
  G33_i = tf.matmul(E_G33 * S_G33_i, E_G33_T)
  G33_sqrt_i = tf.matmul(E_G33 * tf.sqrt(S_G33_i), E_G33_T)
  S_G44, E_G44 = tf.py_func(np.linalg.eigh, [(G44_v + tf.transpose(G44_v)) / 2.0], [tf.float32] * 2)
  S_G44_i = 1.0 / (S_G44 + FLAGS.reg_ratio)
  E_G44_T = tf.transpose(E_G44)
  G44_i = tf.matmul(E_G44 * S_G44_i, E_G44_T)
  G44_sqrt_i = tf.matmul(E_G44 * tf.sqrt(S_G44_i), E_G44_T)
  S_G55, E_G55 = tf.py_func(np.linalg.eigh, [(G55_v + tf.transpose(G55_v)) / 2.0], [tf.float32] * 2)
  S_G55_i = 1.0 / (S_G55 + FLAGS.reg_ratio)
  E_G55_T = tf.transpose(E_G55)
  G55_i = tf.matmul(E_G55 * S_G55_i, E_G55_T)
  G55_sqrt_i = tf.matmul(E_G55 * tf.sqrt(S_G55_i), E_G55_T)
  G66_i_diag = 1.0 / (G66_diag_v + FLAGS.reg_ratio)
  psi_Ab_01 = tf.matmul(A01b_v, A11b_i)
  psi_Ab_12 = tf.matmul(A12b_v, A22b_i)
  psi_Ab_23 = tf.matmul(A23b_v, A33b_i)
  psi_Ab_34 = tf.matmul(A34b_v, A44b_i)
  psi_Ab_45 = tf.matmul(A45b_v, A55b_i)
  psi_G_12 = tf.matmul(G12_v, G22_i)
  psi_G_23 = tf.matmul(G23_v, G33_i)
  psi_G_34 = tf.matmul(G34_v, G44_i)
  psi_G_45 = tf.matmul(G45_v, G55_i)
  psi_G_56 = G56_v * G66_i_diag
  V1 = tf.transpose(grad_W0bi)
  V2 = tf.transpose(grad_W1bi)
  V3 = tf.transpose(grad_W2bi)
  V4 = tf.transpose(grad_W0bg)
  V5 = tf.transpose(grad_W1bg)
  V6 = tf.transpose(grad_W2bg)
  V1 = V1 - tf.matmul(psi_G_12, tf.matmul(V2, tf.transpose(psi_Ab_01)))
  V2 = V2 - tf.matmul(psi_G_23, tf.matmul(V3, tf.transpose(psi_Ab_12)))
  V3 = V3 - tf.matmul(psi_G_34, tf.matmul(V4, tf.transpose(psi_Ab_23)))
  V4 = V4 - tf.matmul(psi_G_45, tf.matmul(V5, tf.transpose(psi_Ab_34))) 
  V5 = V5 - tf.matmul(psi_G_56, tf.matmul(V6, tf.transpose(psi_Ab_45)))
  C01 = tf.matmul(psi_Ab_01, tf.transpose(A01b_v))
  C12 = tf.matmul(psi_Ab_12, tf.transpose(A12b_v))
  C23 = tf.matmul(psi_Ab_23, tf.transpose(A23b_v))
  C34 = tf.matmul(psi_Ab_34, tf.transpose(A34b_v))
  C45 = tf.matmul(psi_Ab_45, tf.transpose(A45b_v))
  D12 = tf.matmul(psi_G_12, tf.transpose(G12_v))
  D23 = tf.matmul(psi_G_23, tf.transpose(G23_v))
  D34 = tf.matmul(psi_G_34, tf.transpose(G34_v))
  D45 = tf.matmul(psi_G_45, tf.transpose(G45_v))
  D56 = tf.matmul(psi_G_56, tf.transpose(G56_v))
  V1 = kronecker_prod_sum_matrix_solve_v3(A00b_sqrt_i, G11_sqrt_i, C01, D12, V1, True)
  V2 = kronecker_prod_sum_matrix_solve_v3(A11b_sqrt_i, G22_sqrt_i, C12, D23, V2)
  V3 = kronecker_prod_sum_matrix_solve_v3(A22b_sqrt_i, G33_sqrt_i, C23, D34, V3)
  V4 = kronecker_prod_sum_matrix_solve_v3(A33b_sqrt_i, G44_sqrt_i, C34, D45, V4)
  V5 = kronecker_prod_sum_matrix_solve_v3(A44b_sqrt_i, G55_sqrt_i, C45, D56, V5)
  V6 = tf.reshape(G66_i_diag, [-1, 1]) * tf.matmul(V6, A55b_i)
  V6 = V6 - tf.matmul(tf.transpose(psi_G_56), tf.matmul(V5, psi_Ab_45))
  V5 = V5 - tf.matmul(tf.transpose(psi_G_45), tf.matmul(V4, psi_Ab_34))
  V4 = V4 - tf.matmul(tf.transpose(psi_G_34), tf.matmul(V3, psi_Ab_23))
  V3 = V3 - tf.matmul(tf.transpose(psi_G_23), tf.matmul(V2, psi_Ab_12))
  V2 = V2 - tf.matmul(tf.transpose(psi_G_12), tf.matmul(V1, psi_Ab_01))
  ema_assign_op = [tf.assign(A00b, A00b_new), tf.assign(A01b, A01b_new), tf.assign(A11b, A11b_new), tf.assign(A12b, A12b_new),\
                tf.assign(A22b, A22b_new), tf.assign(A23b, A23b_new), tf.assign(A33b, A33b_new), tf.assign(A34b, A34b_new),\
                tf.assign(A44b, A44b_new), tf.assign(A45b, A45b_new), tf.assign(A55b, A55b_new),\
                tf.assign(G11, G11_new), tf.assign(G12, G12_new), tf.assign(G22, G22_new), tf.assign(G23, G23_new), tf.assign(G33, G33_new),\
                tf.assign(G34, G34_new), tf.assign(G44, G44_new), tf.assign(G45, G45_new), tf.assign(G55, G55_new), tf.assign(G56, G56_new), tf.assign(G66_diag, G66_diag_new)]
  train_op = optimizer.apply_gradients([(tf.transpose(V1), W0bi), (tf.transpose(V2), W1bi), (tf.transpose(V3), W2bi),\
                                            (tf.transpose(V4), W0bg), (tf.transpose(V5), W1bg), (tf.transpose(V6), W2bg)])
  
  


  np_x_original = tf.placeholder(tf.float32, [None, 28, 28, 1])
  np_x_subsample = tf.nn.avg_pool(np_x_original, ksize=[1, FLAGS.strides, FLAGS.strides, 1], strides=[1, FLAGS.strides, FLAGS.strides, 1], padding='SAME')

  init_op = tf.global_variables_initializer()

  sess = tf.InteractiveSession()
  sess.run(init_op)

  mnist = read_data_sets(FLAGS.data_dir, one_hot=True, validation_size=FLAGS.validation_size)

  total_time_ = 0.0
  if FLAGS.output_file_name_normal != None:
    ostream_ = file(FLAGS.output_file_name_normal, 'w')
  else:
    ostream_ = None

  total_time__ = 0.0
  if FLAGS.output_file_name_q_fisher != None:
    ostream__ = file(FLAGS.output_file_name_q_fisher, 'w')
  else:
    ostream__ = None

  total_time = 0.0
  if FLAGS.output_file_name_r_fisher != None:
    ostream = file(FLAGS.output_file_name_r_fisher, 'w')
  else:
    ostream = None

  np_x_data = mnist.train.images
  np_x_ori = np_x_data.reshape(-1, 28, 28, 1)
  np_x_subsamples = sess.run(np_x_subsample, {np_x_original: np_x_ori})
  np_x_train = (np_x_subsamples > 0.5).astype(np.float32)
  np.random.shuffle(np_x_train)
  np_x_train_fixed = np_x_train[:FLAGS.train_subset_size]

  np_x_data = mnist.test.images
  np_x_ori = np_x_data.reshape(-1, 28, 28, 1)
  np_x_subsamples = sess.run(np_x_subsample, {np_x_original: np_x_ori})
  np_x_test = (np_x_subsamples > 0.5).astype(np.float32)

  if FLAGS.profile:
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

  total_time_t_0 = time.time()
  for i in range(max((FLAGS.n_iterations_normal, FLAGS.n_iterations_q_fisher, FLAGS.n_iterations_r_fisher))):
    np_x_data, _ = mnist.train.next_batch(FLAGS.batch_size)
    np_x_ori = np_x_data.reshape(FLAGS.batch_size, 28, 28, 1)
    np_x_subsamples = sess.run(np_x_subsample, {np_x_original: np_x_ori})
    np_x = (np_x_subsamples > 0.5).astype(np.float32)
    
    if i < FLAGS.n_iterations_normal:
      if FLAGS.profile and (i + 1) % FLAGS.print_every == 0:
        time_start = time.time()
        sess.run(train_op_, feed_dict={x: np_x}, options=options, run_metadata=run_metadata)
        total_time_ += time.time() - time_start
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_step_normal_%d.json' % (i + 1), 'w') as timeline_ostream:
          timeline_ostream.write(chrome_trace)
      else:
        time_start = time.time()
        sess.run(train_op_, {x: np_x})
        total_time_ += time.time() - time_start

    if i < FLAGS.n_iterations_q_fisher:
      if FLAGS.profile and (i + 1) % FLAGS.print_every == 0:
        time_start = time.time()
        sess.run(ema_assign_op__, feed_dict={x: np_x}, options=options, run_metadata=run_metadata)
        sess.run(ema_apply_op__, feed_dict={x: np_x, iter_num: i}, options=options, run_metadata=run_metadata)
        sess.run(train_op__, feed_dict={x: np_x}, options=options, run_metadata=run_metadata)
        total_time__ += time.time() - time_start
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_step_fisher_%d.json' % (i + 1), 'w') as timeline_ostream:
          timeline_ostream.write(chrome_trace)
      else:
        time_start = time.time()
        sess.run(ema_assign_op__, feed_dict={x: np_x})
        sess.run(ema_apply_op__, feed_dict={x: np_x, iter_num: i})
        sess.run(train_op__, feed_dict={x: np_x})
        total_time__ += time.time() - time_start
    
    if i < FLAGS.n_iterations_r_fisher:
      if FLAGS.profile and (i + 1) % FLAGS.print_every == 0:
        time_start = time.time()
        sess.run(ema_assign_op, feed_dict={x: np_x}, options=options, run_metadata=run_metadata)
        sess.run(ema_apply_op, feed_dict={x: np_x, iter_num: i}, options=options, run_metadata=run_metadata)
        sess.run(train_op, feed_dict={x: np_x}, options=options, run_metadata=run_metadata)
        total_time += time.time() - time_start
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_step_fisher_%d.json' % (i + 1), 'w') as timeline_ostream:
          timeline_ostream.write(chrome_trace)
      else:
        time_start = time.time()
        sess.run(ema_assign_op, feed_dict={x: np_x})
        sess.run(ema_apply_op, feed_dict={x: np_x, iter_num: i})
        sess.run(train_op, feed_dict={x: np_x})
        total_time += time.time() - time_start
      
    if (i + 1) % FLAGS.print_every == 0:
      if i < FLAGS.n_iterations_normal:
        np_batch_elbo_ = sess.run(elbo_, {x: np_x})
        np_train_elbo_ = sess.run(elbo_, {x: np_x_train_fixed})
        np_test_elbo_ = sess.run(elbo_, {x: np_x_test})
        to_print_ = 'Iteration: {0:d} Batch NELBO: {1:.3f} Train NELBO: {3:.3f} Test NELBO: {4:.3f} Time elapsed: {2:.2f}'.format(
            i + 1,
            -np_batch_elbo_,
            total_time_,
            -np_train_elbo_,
            -np_test_elbo_)
        if ostream_ == None:
          print to_print_
        else:
          print >> ostream_, to_print_
          ostream_.flush()
      if i < FLAGS.n_iterations_q_fisher:
        np_batch_elbo__ = sess.run(elbo__, {x: np_x})
        np_train_elbo__ = sess.run(elbo__, {x: np_x_train_fixed})
        np_test_elbo__ = sess.run(elbo__, {x: np_x_test})
        to_print__ = 'Iteration: {0:d} Batch NELBO: {1:.3f} Train NELBO: {3:.3f} Test NELBO: {4:.3f} Time elapsed: {2:.2f}'.format(
            i + 1,
            -np_batch_elbo__,
            total_time__,
            -np_train_elbo__,
            -np_test_elbo__)
        if ostream__ == None:
          print to_print__
        else:
          print >> ostream__, to_print__
          ostream__.flush()
      if i < FLAGS.n_iterations_r_fisher:
        np_batch_elbo = sess.run(elbo, {x: np_x})
        np_train_elbo = sess.run(elbo, {x: np_x_train_fixed})
        np_test_elbo = sess.run(elbo, {x: np_x_test})
        to_print = 'Iteration: {0:d} Batch NELBO: {1:.3f} Train NELBO: {3:.3f} Test NELBO: {4:.3f} Time elapsed: {2:.2f}'.format(
            i + 1,
            -np_batch_elbo,
            total_time,
            -np_train_elbo,
            -np_test_elbo)
        if ostream == None:
          print to_print
        else:
          print >> ostream, to_print
          ostream.flush()
      to_print_t = 'Iteration: {0:d} Total elapsed time: {1:.2f}'.format(i + 1, time.time() - total_time_t_0)
      print to_print_t

  if ostream_ != None:
    ostream_.close()

  if ostream__ != None:
    ostream__.close()

  if ostream != None:
    ostream.close()

def main(_):
  if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
  tf.gfile.MakeDirs(FLAGS.logdir)
  train()

if __name__ == '__main__':
  tf.app.run()
