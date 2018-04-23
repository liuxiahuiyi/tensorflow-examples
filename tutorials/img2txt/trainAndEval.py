from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os

from tutorials.img2txt import configuration
from tutorials.img2txt import model as Model
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS


tf.flags.DEFINE_string("train_dir", "./data/mscoco/img2txt_logs",
                       "Directory for saving and loading model checkpoints.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  assert FLAGS.train_dir, "--train_dir is required"

  train_config = configuration.TrainConfig()
  eval_config = configuration.EvalConfig()

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build the TensorFlow graph.
  with tf.Graph().as_default():
    # Build the model.
    train_model = Model.Model(train_config, mode="train")
    train_model.build()
    image_embeddings_var = tf.get_variable(name='image_embeddings_var',
                                           trainable=False,
                                           shape=train_model.image_embeddings.shape,
                                           initializer=tf.zeros_initializer())
    tf.get_variable_scope().reuse_variables()
    eval_model = Model.Model(eval_config, mode="eval")
    eval_model.build()
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    ckpt_state = tf.train.get_checkpoint_state(train_dir)

    saver = tf.train.Saver()
    inception_saver = tf.train.Saver(train_model.inception_variables)
    embedding_config = projector.ProjectorConfig()
    embedding = embedding_config.embeddings.add()
    embedding.tensor_name = 'image_embeddings_var'

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      sess.run(init)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess,coord)

      if ckpt_state and ckpt_state.model_checkpoint_path:
        saver.restore(sess, ckpt_state.model_checkpoint_path)
      else:
        inception_saver.restore(sess, train_dir + '/inception_v3.ckpt')

      summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
      for i in range(10000):
        _, step = sess.run([train_model.train_op, train_model.global_step])
        if step % 10 == 0:
          train_loss, eval_loss, summaries = sess.run(
            [train_model.total_loss, eval_model.total_loss, summary_op])
          print('step: %d, train_loss: %f, eval_loss: %f' % (step, train_loss, eval_loss))
          summary_writer.add_summary(summaries, step)
          summary_writer.flush()
          '''image_embeddings_var.assign(train_model.image_embeddings)
          projector.visualize_embeddings(summary_writer, embedding_config)'''
          saver.save(sess, train_dir + '/model.ckpt', step)

if __name__ == "__main__":
  tf.app.run()