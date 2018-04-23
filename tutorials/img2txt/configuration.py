from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class TrainConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""
    # File pattern of sharded TFRecord file containing SequenceExample protos.
    # Must be provided in training and evaluation modes.
    self.input_file_pattern = '/media/wangyike/7c87e220-304d-43b6-97fd-511ee3036b3a/tensorflow-examples/data/mscoco/tfrecords/train-*-of-00256'

    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.


    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.vocab_size = 10000

    # Batch size.
    self.batch_size = 8

    # Dimensions of Inception v3 input images.
    self.image_height = 299
    self.image_width = 299

    # Scale used to initialize model variables.
    self.initializer_scale = 0.08

    # LSTM input and output dimensionality, respectively.
    self.embedding_size = 512
    self.num_lstm_units = 512

    # If < 1.0, the dropout keep probability applied to LSTM variables.
    self.lstm_dropout_keep_prob = 0.7
    self.grads_clip_norm = 5


class EvalConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    self.input_file_pattern = '/media/wangyike/7c87e220-304d-43b6-97fd-511ee3036b3a/tensorflow-examples/data/mscoco/tfrecords/val-*-of-00004'

    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.


    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.vocab_size = 10000

    # Batch size.
    self.batch_size = 8

    # Dimensions of Inception v3 input images.
    self.image_height = 299
    self.image_width = 299

    # Scale used to initialize model variables.
    self.initializer_scale = 0.08

    # LSTM input and output dimensionality, respectively.
    self.embedding_size = 512
    self.num_lstm_units = 512


class InferenceConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"
    self.initializer_scale = 0.08

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.


    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.vocab_size = 10000

    # Dimensions of Inception v3 input images.
    self.image_height = 299
    self.image_width = 299


    # LSTM input and output dimensionality, respectively.
    self.embedding_size = 512
    self.num_lstm_units = 512
