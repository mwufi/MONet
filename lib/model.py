
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
    endpoints = monet(real_images)
    attn_masks = endpoints['attention_mask']
    log_attn_masks = endpoints['log_attention_mask']
    obj_mask_logits = endpoints['mask_logits']
    obj_masks = endpoints['obj_mask']
    obj_images = endpoints['obj_image']
    obj_latents = endpoints['obj_latent']
    masked_images = tf.stack([tf.multiply(m,i) for m,i in zip(attn_masks, obj_images)], axis=4)
    reconstructed_images = tf.reduce_sum(masked_images, axis=4)

    #### Define the loss
    with tf.name_scope('reconstruction_loss'):
      reconstruction_loss = train_util.reconstruction_loss(
        obj_images, log_attn_masks, real_images,
        background_scale=config['component_scale_background'],
        foreground_scale=config['component_scale_foreground'])

    # compute the VAE latent loss
    with tf.name_scope('vae_loss'):
      cvae_loss = 0
      for latents in obj_latents:
        z_mu, z_log_var = tf.split(latents, num_or_size_splits=2, axis=1)
        cvae_loss += train_util.vae_latent_loss(z_mu, z_log_var)

    # compute loss for VAE output masks
    with tf.name_scope('mask_loss'):
      attention_loss = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        tf.concat(attn_masks, axis=3),
        tf.concat(obj_mask_logits, axis=3),
        axis=3
      ))
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
        tf.summary.image(f'step{i}/{name}', m)
        tf.summary.histogram(f'step{i}/{name}', m[0])

    _log_many('attention_mask', self.attn_masks)
    _log_many('cvae_mask', self.obj_masks)
    _log_many('cvae_image', self.obj_images)

    tf.summary.image('input_images', self.real_images)
    tf.summary.image('inferred_images', self.reconstructed_images)

    tf.summary.scalar('attention_KL', self.attention_loss)
    tf.summary.scalar('normal_vae_KL', self.cvae_loss)
    tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
    tf.summary.scalar('total_loss', self.total_loss)