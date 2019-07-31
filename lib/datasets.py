
"""Module contains a registry of dataset classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from lib import util
import numpy as np
import tensorflow as tf


Counter = collections.Counter


class BaseDataset(object):
  """A base class for reading data from disk."""

  def __init__(self, config):
    self._train_data_path = util.expand_path(config['train_data_path'])

  def provide_dataset(self, batch_size):
    """Provides image dataset"""
    raise NotImplementedError


class ShapesDataset(BaseDataset):
  """A dataset for reading Shapes from a directory with many .npy files."""

  def provide_dataset(self, batch_size):
    pass
