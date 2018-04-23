"""Converts MSCOCO data to TFRecord file format with SequenceExample protos.
The MSCOCO images are expected to reside in JPEG files located in the following
directory structure:
  train_image_dir/COCO_train2014_000000000151.jpg
  train_image_dir/COCO_train2014_000000000260.jpg
  ...
and
  val_image_dir/COCO_val2014_000000000042.jpg
  val_image_dir/COCO_val2014_000000000073.jpg
  ...
The MSCOCO annotations JSON files are expected to reside in train_captions_file
and val_captions_file respectively.
This script converts the combined MSCOCO data into sharded data files consisting
of 256, 4 and 8 TFRecord files, respectively:
  output_dir/train-00000-of-00256
  output_dir/train-00001-of-00256
  ...
  output_dir/train-00255-of-00256
and
  output_dir/val-00000-of-00004
  ...
  output_dir/val-00003-of-00004
and
  output_dir/test-00000-of-00008
  ...
  output_dir/test-00007-of-00008
Each TFRecord file contains ~2300 records. Each record within the TFRecord file
is a serialized SequenceExample proto consisting of precisely one image-caption
pair. Note that each image has multiple captions (usually 5) and therefore each
image is replicated multiple times in the TFRecord files.
The SequenceExample proto contains the following fields:
  context:
    image/image_id: integer MSCOCO image identifier
    image/data: string containing JPEG encoded image in RGB colorspace
  feature_lists:
    image/caption: list of strings containing the (tokenized) caption words
    image/caption_ids: list of integer ids corresponding to the caption words
The captions are tokenized using the NLTK (http://www.nltk.org/) word tokenizer.
The vocabulary of word identifiers is constructed from the sorted list (by
descending frequency) of word tokens in the training set. Only tokens appearing
at least 4 times are considered; all other words get the "unknown" word id.
NOTE: This script will consume around 100GB of disk space because each image
in the MSCOCO dataset is replicated ~5 times (once per caption) in the output.
This is done for two reasons:
  1. In order to better shuffle the training data.
  2. It makes it easier to perform asynchronous preprocessing of each image in
     TensorFlow.
Running this script using 16 threads may take around 1 hour on a HP Z420.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
import imghdr
import json
import os.path
import random
import sys
import threading



import nltk
import numpy as np
import tensorflow as tf

nltk.download('punkt')

tf.flags.DEFINE_string("train_image_dir", "/tmp/train2014/",
                       "Training image directory.")
tf.flags.DEFINE_string("val_image_dir", "/tmp/val2014",
                       "Validation image directory.")

tf.flags.DEFINE_string("train_captions_file", "/tmp/captions_train2014.json",
                       "Training captions JSON file.")
tf.flags.DEFINE_string("val_captions_file", "/tmp/captions_val2014.json",
                       "Validation captions JSON file.")

tf.flags.DEFINE_string("output_dir", "/tmp/", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("vocab_size", 10000,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "/tmp/word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 2,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "caption"])

class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, dict):
    """Initializes the vocabulary.
    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
    self._dict = dict

  def toId(self, word):
    """Returns the integer id of a word string."""
    if word in self._dict:
      return self._dict[word]
    else:
      return self._dict[FLAGS.unknown_word]


def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])



def _to_sequence_example(image, vocab):
  """Builds a SequenceExample proto for an image-caption pair.
  Args:
    image: An ImageMetadata object.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.
  Returns:
    A SequenceExample proto.
  """
  image_type = imghdr.what(image.filename)
  if not image_type in ['jpeg', 'png']:
    return
  with tf.gfile.FastGFile(image.filename, "rb") as f:
    encoded_image = f.read()

  context = tf.train.Features(feature={
      "image/data": _bytes_feature(encoded_image),
  })
  caption_ids = [vocab.toId(word) for word in image.caption]
  feature_lists = tf.train.FeatureLists(feature_list={
      "image/caption_ids": _int64_feature_list(caption_ids)
  })
  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

  return sequence_example


def _process_image_files(thread_index, ranges, shards_per_thread,
                         name, images, vocab):
  """Processes and saves a subset of images as TFRecord files in one thread.
  Args:
    thread_index: Integer thread identifier within [0, len(ranges)].
    ranges: A list of pairs of integers specifying the ranges of the dataset to
      process in parallel.
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """
  # Each thread produces N shards where N = num_shards / num_threads. For
  # instance, if num_shards = 128, and num_threads = 2, then the first thread
  # would produce shards [0, 64).
  counter = 0
  start_shard = thread_index * shards_per_thread
  for shard in range(start_shard, start_shard + shards_per_thread):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    output_filename = "%s-%.5d-of-%.5d" % (name, shard, len(ranges))
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)
    for image in images[ranges[shard][0]:ranges[shard][1]]:

      sequence_example = _to_sequence_example(image, vocab)
      if sequence_example is not None:
        writer.write(sequence_example.SerializeToString())
        counter += 1

      if not counter % 1000:
        print("%s [thread %d]: Processed %d items." %
              (datetime.now(), thread_index, counter))
        sys.stdout.flush()

    writer.close()
    print("%s [thread %d]: Wrote image-caption pairs to %s" %
          (datetime.now(), thread_index, output_file))
    sys.stdout.flush()
  print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
        (datetime.now(), thread_index, counter, shards_per_thread))
  sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
  """Processes a complete data set and saves it as a TFRecord.
  Args:
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """

  # Shuffle the ordering of images. Make the randomization repeatable.
  random.seed(12345)
  random.shuffle(images)

  num_threads = min(num_shards, FLAGS.num_threads)
  shards_per_thread = num_shards // num_threads
  spacing = np.linspace(0, len(images), num_shards + 1).astype(np.int)
  ranges = [] # [ranges[0], ranges[1])
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()
  threads = []
  # Launch a thread for each batch.
  print("Launching %d threads for spacings: %s" % (num_threads, ranges))
  for thread_index in range(num_threads):
    args = (thread_index, ranges, shards_per_thread, name, images, vocab)
    t = threading.Thread(target=_process_image_files, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
        (datetime.now(), len(images), name))


def _create_vocab(words):
  print("Creating vocabulary.")
  counter = [[FLAGS.unknown_word, 0]]
  counter += Counter(words).most_common(FLAGS.vocab_size - 1)
  dictionary = {}
  for w, c in counter:
    dictionary[w] = len(dictionary)
  for word in words:
    if not word in dictionary:
      counter[0][1] += 1

  # Write out the word counts file.
  with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
    f.write("\n".join(["%s %d %d" % (w, dictionary[w], c) for w, c in counter]))
  print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

  vocab = Vocabulary(dictionary)

  return vocab


def _process_caption(caption):
  """Processes a caption string into a list of tonenized words.
  Args:
    caption: A string caption.
  Returns:
    A list of strings; the tokenized caption.
  """
  tokenized_caption = [FLAGS.start_word]
  tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
  tokenized_caption.append(FLAGS.end_word)
  return tokenized_caption


def _load_and_process_metadata(captions_file, image_dir):
  """Loads image metadata from a JSON file and processes the captions.
  Args:
    captions_file: JSON file containing caption annotations.
    image_dir: Directory containing the image files.
  Returns:
    A list of ImageMetadata.
  """
  with tf.gfile.FastGFile(captions_file, "r") as f:
    caption_data = json.load(f)

  # Extract the filenames.
  id_to_filename = [[x["id"], x["file_name"]] for x in caption_data["images"]]

  # Extract the captions. Each image_id is associated with multiple captions.
  id_to_captions = {}
  for annotation in caption_data["annotations"]:
    image_id = annotation["image_id"]
    caption = annotation["caption"]
    id_to_captions.setdefault(image_id, [])
    id_to_captions[image_id].append(_process_caption(caption))
  assert len(id_to_filename) == len(id_to_captions)
  assert set([x[0] for x in id_to_filename]) == set(id_to_captions.keys())
  print("Loaded caption metadata for %d images from %s" %
        (len(id_to_filename), captions_file))

  # Process the captions and combine the data into a list of ImageMetadata.
  print("Processing captions.")
  image_metadata = []
  for image_id, base_filename in id_to_filename:
    filename = os.path.join(image_dir, base_filename)
    for caption in id_to_captions[image_id]:
      image_metadata.append(ImageMetadata(image_id, filename, caption))
  print("Finished processing captions for %d images in %s" %
        (len(id_to_filename), captions_file))

  return image_metadata


def main(unused_argv):
  def _is_valid_num_shards(num_shards):
    """Returns True if num_shards is compatible with FLAGS.num_threads."""
    return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

  assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
  assert _is_valid_num_shards(FLAGS.val_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  # Load image metadata from caption files.
  train_dataset = _load_and_process_metadata(FLAGS.train_captions_file,
                                                    FLAGS.train_image_dir)
  val_dataset = _load_and_process_metadata(FLAGS.val_captions_file,
                                                  FLAGS.val_image_dir)

  # Create vocabulary from the training captions.
  words = []
  for image in train_dataset:
    words += image.caption
  vocab = _create_vocab(words)

  _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
  _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)


if __name__ == "__main__":
  tf.app.run()