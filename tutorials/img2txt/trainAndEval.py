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

  with open('./data/mscoco/words.txt') as f:
    words = f.read()
    words = words.split('\n')
  dictionary = {}
  for word in words:
    info = word.split()
    dictionary[info[0]] = info[1]
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

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
    inception_output_var = tf.get_variable(name='inception_output_var',
                                           trainable=False,
                                           shape=train_model.inception_output.shape,
                                           initializer=tf.zeros_initializer())
    image_embeddings_var = tf.get_variable(name='image_embeddings_var',
                                           trainable=False,
                                           shape=train_model.image_embeddings.shape,
                                           initializer=tf.zeros_initializer())
    image_embeddings_op = tf.assign(image_embeddings_var, train_model.image_embeddings)
    inception_output_op = tf.assign(inception_output_var, train_model.inception_output)
    tf.get_variable_scope().reuse_variables()
    eval_model = Model.Model(eval_config, mode="eval")
    eval_model.build()
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    ckpt_state = tf.train.get_checkpoint_state(train_dir)

    saver = tf.train.Saver()
    inception_saver = tf.train.Saver(train_model.inception_variables)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      sess.run(init)

      if ckpt_state and ckpt_state.model_checkpoint_path:
        saver.restore(sess, ckpt_state.model_checkpoint_path)
      else:
        inception_saver.restore(sess, train_dir + '/inception_v3.ckpt')


      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess,coord)

      summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
      for i in range(10000):
        _, step = sess.run([train_model.train_op, train_model.global_step])
        if step % 10 == 0:
          train_loss, eval_loss, summaries, predictions, _1, _2 = sess.run(
            [train_model.total_loss, eval_model.total_loss, summary_op, train_model.predictions,
                image_embeddings_op, inception_output_op])
          print('step: %d, train_loss: %f, eval_loss: %f' % (step, train_loss, eval_loss))
          tran_predictions = [reverse_dictionary[str(indice)] for indice in predictions]
          print(tran_predictions)
          summary_writer.add_summary(summaries, step)
          summary_writer.flush()
          saver.save(sess, train_dir + '/model.ckpt', step)

if __name__ == "__main__":
  tf.app.run()