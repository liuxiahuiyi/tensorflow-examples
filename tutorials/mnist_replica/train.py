from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
flags.DEFINE_string("data_dir", "./data/mnist/input_data",
                    "Directory for storing mnist data")
flags.DEFINE_string("train_dir", "./data/mnist_replica/log",
                    "Directory for storing train log")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 0,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean("sync_replicas", True,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts","localhost:9009",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:9010,localhost:9011",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")

FLAGS = flags.FLAGS


IMAGE_PIXELS = 28


def main(unused_argv):

  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index =="":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

  #Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")

  # Get the number of workers.
  num_workers = len(worker_spec)

  cluster = tf.train.ClusterSpec({
      "ps": ps_spec,
      "worker": worker_spec})

  if not FLAGS.existing_servers:
    # Not using existing servers. Create an in-process server.
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
      server.join()
      return

  is_chief = (FLAGS.task_index == 0)
  if FLAGS.num_gpus > 0:
    # Avoid gpu allocation conflict: now allocate task_num -> #gpu
    # for each worker in the corresponding machine
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  # The ps use CPU and workers use corresponding GPU
  with tf.Graph().as_default():
    with tf.device(
        tf.train.replica_device_setter(
            worker_device=worker_device,
            ps_device="/job:ps/cpu:0",
            cluster=cluster)):
      job_name = tf.constant(FLAGS.job_name, tf.string)
      task_index = tf.constant(FLAGS.task_index, tf.int64)
      mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
      global_step = tf.train.get_or_create_global_step()
      # Variables of the hidden layer
      hid_w = tf.get_variable(shape=[IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                              initializer=tf.random_normal_initializer(stddev=1.0/IMAGE_PIXELS),
                              name="hid_w")
      hid_b = tf.get_variable(shape=[FLAGS.hidden_units],
                              name="hid_b",
                              initializer=tf.constant_initializer(0.0))

      # Variables of the softmax layer
      sm_w = tf.get_variable(shape=[FLAGS.hidden_units, 10],
                             initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(FLAGS.hidden_units)),
                             name="sm_w")
      sm_b = tf.get_variable(shape=[10],
                             name="sm_b",
                             initializer=tf.constant_initializer(0.0))

      # Ops: located on the worker specified with FLAGS.task_index
      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
      y_ = tf.placeholder(tf.float32, [None, 10])

      hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
      hid = tf.nn.relu(hid_lin)

      logits = tf.nn.xw_plus_b(hid, sm_w, sm_b)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

      opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

      if FLAGS.sync_replicas:
        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=FLAGS.replicas_to_aggregate or num_workers,
            total_num_replicas=num_workers,
            name="mnist_sync_replicas")

      train_op = opt.minimize(loss, global_step=global_step)

      if FLAGS.existing_servers:
        server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
        print("Using existing server at: %s" % server_grpc_url)
      else:
        server_grpc_url = server.target
      sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])
      logging_hook = tf.train.LoggingTensorHook(
        tensors={'job': job_name,
                 'task': task_index,
                 'step': global_step,
                 'train_loss': loss},
        every_n_iter=20)
      stop_hook = tf.train.StopAtStepHook(num_steps=200)
      hooks = [logging_hook, stop_hook]
      if FLAGS.sync_replicas:
        hooks.push(opt.make_session_run_hook(is_chief))
      with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        config=sess_config,
        master=server_grpc_url,
        is_chief=is_chief,
        hooks=hooks) as mon_sess:
        while not mon_sess.should_stop():
          batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
          train_feed = {x: batch_xs, y_: batch_ys}
          mon_sess.run(train_op, feed_dict=train_feed)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()