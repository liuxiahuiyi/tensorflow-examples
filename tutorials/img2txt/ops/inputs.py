from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tutorials.img2txt.ops import image_process

def parse_sequence_example(serialized):
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          'image/data': tf.FixedLenFeature([], dtype=tf.string)
      },
      sequence_features={
          'image/caption_ids': tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

  encoded_image = context['image/data']
  caption = sequence['image/caption_ids']
  return encoded_image, caption


def prefetch_input_data(file_pattern,
                        is_training,
                        batch_size,
                        height,
                        width):
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)
  filename_queue = tf.train.string_input_producer(data_files)
  _, serialized = tf.TFRecordReader().read(filename_queue)
  encoded_image, caption = parse_sequence_example(serialized)
  image = image_process.process_image(encoded_image, is_training, height, width)

  capacity = 100 * batch_size

  return batch_with_dynamic_pad(image, caption, batch_size, capacity)
def batch_with_dynamic_pad(image,
                           caption,
                           batch_size,
                           queue_capacity):
  """Batches input images and captions.
  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.
  Example:
    Actual captions in the batch ('-' denotes padded character):
      [
        [ 1 2 3 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]
    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]
    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]
    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]
  Args:
    images_and_captions: A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.
  Returns:
    images: A Tensor of shape [batch_size, height, width, channels].
    input_seqs: An int32 Tensor of shape [batch_size, padded_length].
    target_seqs: An int32 Tensor of shape [batch_size, padded_length].
    mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
  """
  num_threads = 2
  caption_length = tf.shape(caption)[0]
  input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
  input_seq = tf.slice(caption, [0], input_length)
  target_seq = tf.slice(caption, [1], input_length)
  mask = tf.ones(input_length, dtype=tf.int32)
  images, input_seqs, target_seqs, masks = tf.train.batch(
      [image, input_seq, target_seq, mask],
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=queue_capacity,
      dynamic_pad=True)

  # Display the training images in the visualizer.
  return images, input_seqs, target_seqs, masks
