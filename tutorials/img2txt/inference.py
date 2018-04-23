from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tutorials.img2txt import configuration
from tutorials.img2txt import model as Model
from tutorials.img2txt.ops import image_process
import matplotlib.pyplot as plt
FLAGS = tf.app.flags.FLAGS


tf.flags.DEFINE_string("model_dir", "./data/mscoco/img2txt_logs",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("inception_file", "./data/mscoco/img2txt_logs/inception_v3.ckpt",
                       "File for saving and loading model inception.")
tf.flags.DEFINE_string("img_path", "./data/mscoco/raw-data/train2014/COCO_train2014_000000237327.jpg",
                       "Image path.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

  inference_config = configuration.InferenceConfig()
  with open('./data/mscoco/words.txt') as f:
    words = f.read()
    words = words.split('\n')
  dictionary = {}
  for word in words:
    info = word.split()
    dictionary[info[0]] = info[1]
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  # Build the TensorFlow graph.
  with tf.Graph().as_default():
    # Build the model.
    model = Model.Model(inference_config, mode="inference")
    model.build()
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      raise Exception('No model found')
    saver = tf.train.Saver()
    inception_saver = tf.train.Saver(model.inception_variables)

    encoded_image = tf.gfile.FastGFile(FLAGS.img_path, "rb").read()
    image = image_process.process_image(encoded_image, False, inference_config.image_height,
                          inference_config.image_width)
    image = tf.expand_dims(image, 0)
    with tf.Session() as sess:
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      inception_saver.restore(sess, FLAGS.inception_file)
      image_value = sess.run(image)
      plt.imshow(image_value[0])
      plt.show()
      initial_state_concat = sess.run(model.initial_state_concat,
                                     feed_dict={model.images:image_value})
      last_state_concat = initial_state_concat
      new_state_concat = None
      prediction = [2]
      sentence = ''
      while prediction[0] != 3:
        prediction, new_state_concat = sess.run([model.prediction,model.new_state_concat],
      	                               feed_dict={
      	                                           model.last_state_concat:last_state_concat,
      	                                           model.input_seqs: prediction,
      	                                         })
        last_state_concat = new_state_concat
        if prediction[0] != 3:
          sentence += (reverse_dictionary[str(prediction[0])] + ' ')
      print(sentence)

if __name__ == "__main__":
  tf.app.run()