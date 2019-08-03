"""Train a MONet model.

Example usage: (from base directory):
>>> python monet_train.py

to use a config of hyperparameters:
>>> python monet_train.py --config mini

to use a config of hyperparameters ~and~ manual params
>>> python monet_train.py --config mini \
>>> --hparams='{"train_data_path": "/path/to/training.tfrecord"}'

List of hyperparameters can be found in model.py
Trains in a couple of days on a single V100 GPU.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import importlib
import json
import os

from absl import logging
import absl.flags
from lib import data_helpers
from lib import data_normalizer
from lib import flags as lib_flags
from lib import model as lib_model
from lib import train_util
from lib import util
import tensorflow as tf


absl.flags.DEFINE_string('hparams', '{}', 'Flags dict as JSON string.')
absl.flags.DEFINE_string('config', '', 'Name of config module.')
FLAGS = absl.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def init_data_normalizer(config):
  """Initializes data normalizer."""
  normalizer = data_normalizer.registry[config['data_normalizer']](config)
  if normalizer.exists():
    return

  if config['task'] == 0:
    tf.reset_default_graph()
    data_helper = data_helpers.registry[config['data_type']](config)
    real_images, _ = data_helper.provide_data(batch_size=10)

    # Save normalizer.
    # Note if normalizer has been saved, save() is no-op. To regenerate the
    # normalizer, delete the normalizer file in train_root_dir/assets
    normalizer.save(real_images)
  else:
    while not normalizer.exists():
      time.sleep(5)


def run(config):
  """Entry point to run training"""
  init_data_normalizer(config)
  batch_size = config['batch_size']
  tf.reset_default_graph()
  with tf.device(tf.train.replica_device_setter(config['ps_tasks'])):
    model = lib_model.Model(batch_size, config)
    model.add_summaries()
    print('Variables:')
    for v in tf.global_variables():
      print('\t', v.name, v.get_shape().as_list())
    logging.info('Calling train method!')
    train_util.train(model, **config)


def main(unused_argv):
  absl.flags.FLAGS.alsologtostderr = True
  # Set hyperparams from json args and defaults
  flags = lib_flags.Flags()
  # Config hparams
  if FLAGS.config:
    config_module = importlib.import_module(
        'configs.{}'.format(FLAGS.config))
    flags.load(config_module.hparams)
  # Command line hparams
  flags.load_json(FLAGS.hparams)
  # Set default flags
  lib_model.set_flags(flags)

  print('Flags:')
  flags.print_values()

  # Create training directory
  flags['train_root_dir'] = util.expand_path(flags['train_root_dir'])
  if not tf.io.gfile.exists(flags['train_root_dir']):
    tf.io.gfile.mkdir(flags['train_root_dir'])

  # Save the flags for later
  fname = os.path.join(flags['train_root_dir'], 'experiment.json')
  with  tf.io.gfile.GFile(fname, 'w') as f:
    json.dump(flags, f)

  # Run training
  run(flags)


if __name__ == '__main__':
  tf.app.run(main)