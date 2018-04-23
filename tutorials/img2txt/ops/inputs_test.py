import tensorflow as tf
from tutorials.img2txt.ops import inputs

with tf.Graph().as_default():
  a, b, c, d = inputs.prefetch_input_data('/media/wangyike/7c87e220-304d-43b6-97fd-511ee3036b3a/tensorflow-examples/data/mscoco/tfrecords/train-*-of-00256',
    	                                  True, 5, 299, 299)
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    sess.run(init)
    images, input_seqs, target_seqs, masks = sess.run([a, b, c, d])
    print(input_seqs, target_seqs)
    coord.request_stop()
    coord.join(threads)