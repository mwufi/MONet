
"""Train MONet
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import logging
from magenta.models.gansynth.lib import networks
import numpy as np
import tensorflow as tf


def make_train_sub_dir(**kwargs):
  """Returns the log directory for training stage `stage_id`."""
  return os.path.join(kwargs['train_root_dir'], 'stage_{:05d}'.format(0))


def define_train_ops(loss, **config):
  if config['optimizer'] == 'RMSProp':
    optimizer = tf.train.RMSPropOptimizer(config['lr'])
    global_step = tf.train.get_or_create_global_step()
    train_ops = optimizer.minimize(loss, global_step=global_step)
    return train_ops, optimizer
  else:
    raise NotImplementedError


def reconstruction_loss(x_reconstructed, x_true):
  """Just use MSE because we have a gaussian decoder"""
  return tf.losses.mean_squared_error(x_reconstructed, x_true)


def gaussian_kl(z_mean, z_log_variance):
  """Returns the KL from N(0,1)"""
  kl_div_loss = 1 + z_log_variance - tf.square(z_mean) - tf.exp(z_log_variance)
  kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
  return kl_div_loss


class ThrowUpDebugHook(tf.train.SessionRunHook):
  """Prints summary statistics of all tf variables."""

  def __init__(self):
    super(ThrowUpDebugHook, self).__init__()
    self._fetches = [v for v in tf.global_variables()]

  def before_run(self, _):
    return tf.train.SessionRunArgs(self._fetches)

  def after_run(self, _, vals):
    print('=============')
    print('Weight stats:')
    for v, r in zip(self._fetches, vals.results):
      print('\t', v.name, np.min(r), np.mean(r), np.max(r), r.shape)
    print('=============')


def train(model, **kwargs):
  """Train MONet

  Args:
    model: A model object having all the information we need, e.g. the
      return of build_model()
    **kwargs: A dictionary of:
      'train_root_dir': A string of root directory of training logs
      'master': name of TensorFlow master to use
      'task': the task ID. this value is used when training with multiple workers
      'save_summaries_num_images': save summaries in this number of images
      'debug_hook': whether to attach the debug hook to the training sess

  Returns:
    None.
  """
  logging.info("Starting training!")
  logdir = make_train_sub_dir( **kwargs)
  global_step = tf.train.get_or_create_global_step()

  logging_dict = {
    'global_step': global_step,
    'loss': model.total_loss
  }
  hooks = [
    tf.train.LoggingTensorHook(
        logging_dict, every_n_iter=kwargs['save_summaries_num_images']),
    tf.train.StepCounterHook(
      output_dir=logdir, every_n_steps=kwargs['save_summaries_num_images'])
  ]
  if kwargs['debug_hook']:
    hooks.append(ThrowUpDebugHook)

  scaffold = tf.train.Scaffold(
    saver=tf.train.Saver(
      max_to_keep=kwargs['checkpoints_to_keep'],
      keep_checkpoint_every_n_hours=1))

  # Run the training loop
  tf.contrib.training.train(
    model.train_op,
    logdir,
    master=kwargs['master'],
    is_chief=(kwargs['task'] == 0),
    scaffold=scaffold,
    hooks=hooks,
    save_checkpoint_secs=600,
    save_summaries_steps=(kwargs['save_summaries_num_images']),
  )

