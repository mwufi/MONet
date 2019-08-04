
"""GANSynth Model class definition.

Exposes external API for generating samples and evaluation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib import data_helpers
import tensorflow_probability as tfp
tfd = tfp.distributions
from lib import networks
from lib import train_util
import tensorflow as tf


def set_flags(flags):
  """Set default hyperparameters."""
  flags.set_if_empty('train_data_path', 'data')

  ### Logging
  flags.set_if_empty('train_root_dir', 'experiments')
  flags.set_if_empty('save_summaries_num_images', 10)

  ### Data
  flags.set_if_empty('data_type', 'image')
  flags.set_if_empty('data_normalizer', 'none') # TODO: Add another normalizer!

  ### Distributed Training ###
  flags.set_if_empty('master', '')  # Name of the TensorFlow master to use
  # The number of parameter servers. If the value is 0, then the parameters
  # are handled locally by the worker
  flags.set_if_empty('ps_tasks', 0)
  # The Task ID. This value is used when training with multiple workers to
  # identify each worker
  flags.set_if_empty('task', 0)

  ### Debugging ###
  flags.set_if_empty('debug_hook', False)


class Model(object):
  """MONet model"""

  def __init__(self, batch_size, config):
    """ Builds a model that you can train!

    Args:
      batch_size: (int) Build graph with fixed batch size.
      config: (dict) All the global state.
    """
    data_helper = data_helpers.registry[config['data_type']](config)
    real_images = data_helper.provide_data(batch_size)

    #### Define the model
    monet = networks.MONet(**config)
    reconstructed_images, endpoints = monet(real_images)
    attn_masks = endpoints['attention_mask']
    obj_masks = endpoints['obj_mask']
    obj_images = endpoints['obj_image']
    obj_latents = endpoints['obj_latent']

    #### Define the loss

    # using MSE loss for now
    reconstructed_images = tf.reduce_sum(tf.stack(obj_images, axis=0), axis=0)
    reconstruction_loss = tf.compat.v1.losses.mean_squared_error(reconstructed_images, real_images)

    # compute the VAE latent loss
    cvae_loss = 0
    for latents in obj_latents:
      z_mu, z_log_sigma = tf.split(latents, num_or_size_splits=2, axis=1)
      cvae_loss += tf.reduce_sum(train_util.gaussian_kl(z_mu, z_log_sigma))

    # compute loss for VAE output masks
    attention_prior = tfp.distributions.Categorical(probs=tf.concat(attn_masks, axis=3))
    attention_guess = tfp.distributions.Categorical(probs=tf.nn.softmax(tf.concat(obj_masks, axis=3)))
    attention_loss = tf.reduce_sum(tfp.distributions.kl_divergence(attention_prior, attention_guess))
    loss = reconstruction_loss \
           + config['beta'] * cvae_loss \
           + config['gamma'] * attention_loss

    ##### Define train ops.
    train_op, optimizer = train_util.define_train_ops(
      loss, **config
    )

    ##### Add variables as properties
    self.config = config
    self.data_helper = data_helper
    self.real_images = real_images
    self.reconstructed_images = reconstructed_images
    self.attn_masks = attn_masks
    self.obj_masks = obj_masks
    self.obj_images = obj_images
    self.obj_latent = obj_latents
    self.train_op = train_op
    self.attention_loss = attention_loss
    self.cvae_loss = cvae_loss
    self.reconstruction_loss = reconstruction_loss
    self.total_loss = loss
    self.monet_infer = monet


  def add_summaries(self):
    """Adds model summaries."""
    config = self.config
    data_helper = self.data_helper

    def _log_many(name, images):
      for i, m in enumerate(images):
        tf.summary.image(f'step_{i}/{name}', m)
        tf.summary.histogram(f'step_{i}/{name}', m[0])

    _log_many('attention_mask', self.attn_masks)
    _log_many('cvae_mask', self.obj_masks)
    _log_many('cvae_image', self.obj_images)

    tf.summary.image('input_images', self.real_images)
    tf.summary.image('inferred_images', self.reconstructed_images)

    tf.summary.scalar('attention_KL', self.attention_loss)
    tf.summary.scalar('normal_vae_KL', self.cvae_loss)
    tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
    tf.summary.scalar('total_loss', self.total_loss)