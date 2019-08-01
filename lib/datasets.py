
"""Module contains a registry of dataset classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
from lib import util
import numpy as np
import tensorflow as tf
from tensorflow.io import FixedLenFeature


Counter = collections.Counter


class BaseDataset(object):
  """A base class for reading data from disk."""

  def __init__(self, config):
    self._train_data_path = util.expand_path(config['train_data_path'])

  def provide_dataset(self, batch_size):
    """Provides image dataset"""
    raise NotImplementedError

class ShapesTFRecordDataset(BaseDataset):
  """A dataset for reading TFRecords from a directory"""

  def _get_dataset_from_path(self):
    filenames = tf.io.gfile.listdir(self._train_data_path)
    filenames = [os.path.join(self._train_data_path, x) for x in filenames]
    d = tf.data.TFRecordDataset(filenames)
    d = d.repeat()
    return d

  def preprocess(self, image):
    """Preprocess a single image in [height, width, depth] layout."""
    return image

  def provide_dataset(self):
    dataset = self._get_dataset_from_path()

    def _parse_image(record):
      features = {
        'n_objects': FixedLenFeature([1], dtype=tf.int64),
        'image': FixedLenFeature([], dtype=tf.string)
      }
      example = tf.io.parse_single_example(record, features)
      n_objects, image = example['n_objects'], example['image']
      image = tf.decode_raw(image, tf.uint8)

      DEPTH = 3
      HEIGHT = WIDTH = 128
      image.set_shape([DEPTH * HEIGHT * WIDTH])

      # Reshape from [depth * height * width] to [height, width, depth].
      image = tf.cast(
        tf.reshape(image, [HEIGHT, WIDTH, DEPTH]),
        tf.float32)

      # Custom preprocessing.
      image = self.preprocess(image)
      print(n_objects)
      return n_objects, image

    dataset = dataset.map(_parse_image, num_parallel_calls=4)

    # Filter for only 5-object data
    # TODO Currently this line causes an error :P
    # dataset = dataset.filter(lambda n, i: tf.math.equal(n, 5))
    dataset = dataset.map(lambda n, i: i)
    return dataset



class ShapesNumpyDataset(BaseDataset):
  """A dataset for reading Shapes from a directory with many .npy files."""

  def _get_dataset_from_path(self):
    d = tf.data.Dataset.list_files(self._train_data_path)
    d = d.repeat()
    return d

  def provide_dataset(self):
    def _parse_npy(file):
      array = np.load(file)
      return array

    dataset = self._get_dataset_from_path()
    dataset = dataset.map(_parse_npy, num_parallel_calls=4)
    return dataset


registry = {
  'shapes_numpy': ShapesNumpyDataset,
  'shapes': ShapesTFRecordDataset
}