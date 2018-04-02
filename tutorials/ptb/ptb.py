"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from tutorials.ptb import reader
from tutorials.ptb import util


flags = tf.flags

flags.DEFINE_string('data_path', './data/ptb/ptbdata',
                    'Where the training/test data is stored.')
flags.DEFINE_string('save_path', './data/ptb/logs',
                    'Model output directory.')
FLAGS = flags.FLAGS

INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.7
class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps
    self.epoch_size = ((len(data) // config.batch_size) - 1) // config.num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, self.batch_size, self.num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    self.global_step = tf.train.get_or_create_global_step()
    size = config.hidden_size
    vocab_size = config.vocab_size

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=tf.float32)
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph(inputs, config, is_training)

    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  tf.train.get_or_create_global_step(),
                                  config.decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == 'basic':
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == 'block':
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, tf.float32)
    state = self._initial_state
    # Simplified version of tf.nn.static_rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    # outputs, state = tf.nn.static_rnn(cell, inputs,
    #                                   initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

class Config(object):
  init_scale = 0.05
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  decay_steps = 50
  batch_size = 20
  vocab_size = 10000
  rnn_mode = 'block'

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  config = Config()
  train_data, valid_data, word_to_id = reader.ptb_raw_data(FLAGS.data_path, config.vocab_size)

  with tf.name_scope("Train"):
    train_input = PTBInput(config=config, data=train_data, name="TrainInput")
    with tf.variable_scope("Model", reuse=None):
      m = PTBModel(is_training=True, config=config, input_=train_input)
    tf.summary.scalar("Training Loss", m.cost)
    tf.summary.scalar("Learning Rate", m.lr)
  with tf.name_scope("Valid"):
    valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
    with tf.variable_scope("Model", reuse=True):
      mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
    tf.summary.scalar("Validation Loss", mvalid.cost)
  summary_hook = tf.train.SummarySaverHook(
    save_steps=10,
    output_dir=FLAGS.save_path,
    summary_op=tf.summary.merge_all())
  logging_hook = tf.train.LoggingTensorHook(
    tensors={'step': m.global_step,
             'train_loss': m.cost,
             'valiade_loss': mvalid.cost,
             'lr': m.lr},
    every_n_iter=10)
  with tf.train.MonitoredTrainingSession(
    checkpoint_dir=FLAGS.save_path,
    hooks=[logging_hook],
    chief_only_hooks=[summary_hook],
    # Since we provide a SummarySaverHook, we need to disable default
    # SummarySaverHook. To do that we set save_summaries_steps to 0.
    save_summaries_steps=0,
    config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run([m.train_op, mvalid.cost])


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()