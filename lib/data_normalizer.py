
"""Data normalizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os

from absl import logging
import numpy as np
import tensorflow as tf


def _range_normalizer(x, margin):
  x = x.flatten()
  min_x = np.min(x)
  max_x = np.max(x)
  a = margin * (2.0 / (max_x - min_x))
  b = margin * (-2.0 * min_x / (max_x - min_x) - 1.0)
  return a, b


class DataNormalizer(object):
  """A class to normalize data."""

  def __init__(self, config, file_name):
    self._work_dir = os.path.join(config['train_root_dir'], 'assets')
    self._margin = config['normalizer_margin']
    self._path = os.path.join(self._work_dir, file_name)
    self._done_path = os.path.join(self._work_dir, 'DONE_' + file_name)
    self._num_examples = config['normalizer_num_examples']

  def _run_data(self, data):
    """Runs data in session to get data_np."""
    if data is None:
      return None
    data_np = []
    count_examples = 0
    with tf.MonitoredTrainingSession() as sess:
      while count_examples < self._num_examples:
        out = sess.run(data)
        data_np.append(out)
        count_examples += out.shape[0]
    data_np = np.concatenate(data_np, axis=0)
    return data_np

  def compute(self, data_np):
    """Computes normalizer."""
    raise NotImplementedError

  def exists(self):
    return tf.gfile.Exists(self._done_path)

  def save(self, data):
    """Computes and saves normalizer."""
    if self.exists():
      logging.info('Skip save() as %s already exists', self._done_path)
      return
    data_np = self._run_data(data)
    normalizer = self.compute(data_np)
    logging.info('Save normalizer to %s', self._path)
    bytes_io = io.BytesIO()
    np.savez(bytes_io, normalizer=normalizer)
    if not tf.gfile.Exists(self._work_dir):
      tf.gfile.MakeDirs(self._work_dir)
    with tf.gfile.Open(self._path, 'wb') as f:
      f.write(bytes_io.getvalue())
    with tf.gfile.Open(self._done_path, 'w') as f:
      f.write('')
    return normalizer

  def load(self):
    """Loads normalizer."""
    logging.info('Load data from %s', self._path)
    with tf.gfile.Open(self._path, 'rb') as f:
      result = np.load(f)
      return result['normalizer']

  def normalize_op(self, x):
    raise NotImplementedError

  def denormalize_op(self, x):
    raise NotImplementedError


class NoneNormalizer(object):
  """A dummy class that does not normalize data."""

  def __init__(self, unused_config=None):
    pass

  def save(self, data):
    pass

  def load(self):
    pass

  def exists(self):
    return True

  def normalize_op(self, x):
    return x

  def denormalize_op(self, x):
    return x

registry = {
  'none': NoneNormalizer
}