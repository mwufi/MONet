
import tensorflow as tf
from tensorflow.compat.v1 import GraphKeys
from lib.datasets import registry

class DataHelper(object):
  """A class for querying data"""
  def __init__(self, config):
    self._config = config
    self._dataset = registry[config['dataset_name']](config)

  def _map_fn(self):
    """Create a mapping function for the dataset"""
    raise NotImplementedError

  def provide_data(self, batch_size):
    """Returns a batch of images"""
    with tf.name_scope('inputs'):
      with tf.device('/cpu:0'):
        dataset = self._dataset.provide_dataset()
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(self._map_fn, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        tf.compat.v1.add_to_collection(GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

        data = iterator.get_next()
        data.set_shape([batch_size, None, None, None])
        return data

class DataImagesHelper(DataHelper):
  """A data helper for raw images"""
  def _map_fn(self, images):
    return images