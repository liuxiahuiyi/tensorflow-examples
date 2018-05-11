from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tutorials.img2txt import configuration
from tutorials.img2txt import model as Model
from PIL import Image
import matplotlib.pyplot as plt
FLAGS = tf.app.flags.FLAGS


tf.flags.DEFINE_string("model_dir", "./data/mscoco/img2txt_logs",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("img_path", "./data/mscoco/raw-data/val2014/COCO_val2014_000000285844.jpg",
                       "Image path.")
tf.flags.DEFINE_string("summary_dir", "./data/mscoco/summary_dir",
                       "Directory for saving and loading model checkpoints.")



def main(unused_argv):
  img=Image.open(FLAGS.img_path)
  img.show()

  with open('./data/mscoco/words.txt') as f:
    words = f.read()
    words = words.split('\n')
  dictionary = {}
  for word in words:
    info = word.split()
    dictionary[info[0]] = info[1]
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  inference_config = configuration.InferenceConfig()
  # Build the TensorFlow graph.
  with tf.Graph().as_default():
    # Build the model.
    model = Model.Model(inference_config, mode="inference")
    model.build()
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      raise Exception('No model found')
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()

    encoded_image = tf.gfile.FastGFile(FLAGS.img_path, "rb").read()

    with tf.Session() as sess:

      saver.restore(sess, ckpt_state.model_checkpoint_path)
      initial_state = sess.run(model.initial_state,
                                      feed_dict={
                                                  model.image_placeholder:encoded_image,
                                                })
      last_state = initial_state
      predictions = [2]
      sentence = ''
      '''summaries = sess.run(summary_op, feed_dict={
                                                   model.image_placeholder:encoded_image,
                                                   model.last_state_concat:last_state_concat,
                                                   model.input_seqs: predictions,
                                                 })'''
      while predictions[0] != 3:
        predictions, last_state = sess.run([model.predictions,model.new_state],
      	                                feed_dict={
      	                                            model.last_state:last_state,
      	                                            model.input_seqs: predictions,
      	                                          })
        print(last_state)
        if predictions[0] != 3:
          sentence += (reverse_dictionary[str(predictions[0])] + ' ')
      print(sentence)
      '''summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
      summary_writer.add_summary(summaries)
      summary_writer.flush()'''

if __name__ == "__main__":
  tf.app.run()