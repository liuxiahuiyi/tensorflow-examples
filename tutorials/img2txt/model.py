from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tutorials.img2txt import image_embedding
from tutorials.img2txt.ops import inputs


class Model(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.
  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode):
    """Basic setup.
    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      trainable: Whether the inception submodel variables are trainable.
    """
    assert mode in ['train', 'eval', 'inference']
    self.config = config
    self.mode = mode

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.masks = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Global step Tensor.
    self.global_step = tf.train.get_or_create_global_step()
    self.inception_variables = []

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.
    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == 'inference':
      self.images = tf.placeholder(tf.float32,
        shape=[1,self.config.image_height,self.config.image_width,3])
      self.input_seqs = tf.placeholder(tf.int64, shape=[1])
      return
    images, input_seqs, target_seqs, masks = inputs.prefetch_input_data(self.config.input_file_pattern,
                                                                        self.is_training(),
                                                                        self.config.batch_size,
                                                                        self.config.image_height,
                                                                        self.config.image_width)

    self.images, self.input_seqs, self.target_seqs, self.masks = images, input_seqs, target_seqs, masks
  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings.
    Inputs:
      self.images
    Outputs:
      self.image_embeddings
    """
    inception_output = image_embedding.inception_v3(
        self.images,
        is_training=self.is_training())
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      self.image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")

  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.
    Inputs:
      self.input_seqs
    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      self.seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)


  def build_model(self):
    """Builds the model.
    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
    if self.mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)

    with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      zero_state = lstm_cell.zero_state(
          batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
      _, initial_state = lstm_cell(self.image_embeddings, zero_state)

      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()
      if self.mode == 'inference':
        self.initial_state_concat = tf.concat(axis=1, values=initial_state)
        self.last_state_concat = tf.placeholder(tf.float32, [1, sum(lstm_cell.state_size)])
        last_state = tf.split(value=self.last_state_concat, num_or_size_splits=2, axis=1)
        lstm_outputs, new_state = lstm_cell(self.seq_embeddings, state=last_state)
        self.new_state_concat = tf.concat(axis=1, values=new_state)
      else:
        # Run the batch of sequence embeddings through the LSTM.
        sequence_length = tf.reduce_sum(self.masks, 1)
        lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=self.seq_embeddings,
                                            sequence_length=sequence_length,
                                            initial_state=initial_state,
                                            dtype=tf.float32,
                                            scope=lstm_scope)

    # Stack batches vertically.
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    with tf.variable_scope("logits") as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=lstm_outputs,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_scope)
    if self.mode == 'inference':
      self.prediction = tf.argmax(logits, axis=1)
      return
    targets = tf.reshape(self.target_seqs, [-1])
    weights = tf.to_float(tf.reshape(self.masks, [-1]))
    # Compute losses.
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                            logits=logits)
    batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                        tf.reduce_sum(weights),
                        name="batch_loss")
    tf.losses.add_loss(batch_loss)
    total_loss = tf.losses.get_total_loss()
    self.total_loss = total_loss
    self.target_cross_entropy_losses = losses  # Used in evaluation.
    self.target_cross_entropy_loss_weights = weights  # Used in evaluation.
    self.prediction = tf.argmax(logits, axis=1)
    # Add summaries.
    if self.is_training():
      tf.summary.scalar("losses/batch_loss", batch_loss)
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)
      for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
        tf.summary.histogram("moving_average/" + var.op.name, var)
      tf.summary.histogram('image_embeddings', self.image_embeddings)
      tf.summary.histogram('seq_embeddings', self.seq_embeddings)
      tf.summary.histogram('predictions', tf.multiply(tf.to_float(self.prediction), weights))

  def build_train_op(self):
    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.total_loss, trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, self.config.grads_clip_norm)
    optimizer = tf.train.AdamOptimizer()
    apply_op = optimizer.apply_gradients(
      zip(grads, trainable_variables),
      global_step=self.global_step, name='train_step')
    train_ops = [apply_op] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    self.train_op = tf.group(*train_ops)

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_model()
    if self.is_training():
      self.build_train_op()
