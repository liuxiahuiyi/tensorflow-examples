from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
slim = tf.contrib.slim

def inception_v3(images,
                 is_training,
                 weight_decay=0.0004,
                 stddev=0.1,
                 dropout_keep_prob=0.8):

  if is_training:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  batch_norm_params = {
      "is_training": is_training,
      "trainable": is_training,
      # Decay for the moving averages.
      "decay": 0.9997,
      # Epsilon to prevent 0s in variance.
      "epsilon": 0.001,
      # Collection containing the moving mean and moving variance.
      "variables_collections": {
          "beta": None,
          "gamma": None,
          "moving_mean": ["moving_vars"],
          "moving_variance": ["moving_vars"],
      }
  }
  with tf.variable_scope("InceptionV3", values=[images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=is_training):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception_v3_base(images, scope=scope)
        shape = net.get_shape()
        net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
        net = slim.dropout(
            net,
            keep_prob=dropout_keep_prob,
            is_training=is_training,
            scope="dropout")
        net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if is_training:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)
  return net