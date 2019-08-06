
"""GANSynth Model class definition.

Exposes external API for generating samples and evaluation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib import data_helpers
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
  """Base model"""

  def __init__(self, batch_size, config):
    """ Builds a model that you can train!

    Args:
      batch_size: (int) Build graph with fixed batch size.
      config: (dict) All the global state.
    """
    data_helper = data_helpers.registry[config['data_type']](config)
    real_images = data_helper.provide_data(batch_size)

    self.config = config
    self.data_helper = data_helper
    self.real_images = data_helper.provide_data(batch_size)

    self.build_model(config)
    self.define_train_op(config)

  def build_model(self, config):
    raise NotImplementedError

  def define_train_op(self, config):
    """Returns a train_op"""
    raise NotImplementedError

  def add_summaries(self):
    raise NotImplementedError

class VAEModel(Model):
  """The component VAE by itself"""
  def build_model(self, config):
    vae = networks.ComponentVAE(**config)

    batch_size, h, w, c = self.real_images.shape
    full_mask = tf.zeros((batch_size, h, w, 1))
    object_logits, mask_logits, latents = vae(tf.concat([self.real_images, full_mask], axis=3))

    self.attn_mask = [full_mask]
    self.obj_mask = [tf.nn.sigmoid(mask_logits)]
    self.obj_image = [tf.nn.sigmoid(object_logits)]
    self.latents = [latents]

  def define_train_op(self, config):
    with tf.name_scope('reconstruction_loss'):
      reconstruction_loss = train_util.reconstruction_loss(
        self.obj_image, self.attn_mask, self.real_images,
        background_scale=config['component_scale_background'],
        foreground_scale=config['component_scale_foreground'])

    # compute the VAE latent loss
    with tf.name_scope('vae_loss'):
      cvae_loss = 0
      for latents in self.latents:
        z_mu, z_log_var = tf.split(latents, num_or_size_splits=2, axis=1)
        cvae_loss += train_util.vae_latent_loss(z_mu, z_log_var)

    self.reconstruction_loss = reconstruction_loss
    self.cvae_loss = cvae_loss
    self.total_loss = reconstruction_loss + cvae_loss

    train_op, optimizer = train_util.define_train_ops(
      self.total_loss, **config
    )
    self.train_op = train_op

  def add_summaries(self):
    """Adds model summaries."""

    def _log_many(name, images):
      for i, m in enumerate(images):
        tf.summary.image(f'step{i}/{name}', m)
        tf.summary.histogram(f'step{i}/{name}', m[0])

    _log_many('cvae_mask', self.obj_mask)
    _log_many('cvae_image', self.obj_image)

    tf.summary.image('input_images', self.real_images)

    tf.summary.scalar('normal_vae_KL', self.cvae_loss)
    tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
    tf.summary.scalar('total_loss', self.total_loss)


class MonetModel(Model):
  """Monet model"""

  def build_model(self, config):
    #### Define the model
    monet = networks.MONet(**config)
    endpoints = monet(self.real_images)
    attn_masks = endpoints['attention_mask']
    log_attn_masks = endpoints['log_attention_mask']
    obj_mask_logits = endpoints['mask_logits']
    obj_masks = endpoints['obj_mask']
    obj_images = endpoints['obj_image']
    obj_latents = endpoints['obj_latent']

    with tf.name_scope('inferred_image'):
      masked_images = tf.stack([tf.multiply(m,i) for m,i in zip(attn_masks, obj_images)], axis=4)
      reconstructed_images = tf.reduce_sum(masked_images, axis=4)

    #### Define the loss
    with tf.name_scope('reconstruction_loss'):
      reconstruction_loss = train_util.reconstruction_loss(
        obj_images, log_attn_masks, self.real_images,
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
      attention_loss = tf.reduce_sum(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
        tf.concat(attn_masks, axis=3),
        tf.concat(obj_mask_logits, axis=3),
        axis=3
      ))
    loss = reconstruction_loss \
           + config['beta'] * cvae_loss \
           + config['gamma'] * attention_loss

    ##### Add variables as properties
    self.reconstructed_images = reconstructed_images
    self.attn_masks = attn_masks
    self.obj_masks = obj_masks
    self.obj_images = obj_images
    self.obj_latent = obj_latents
    self.attention_loss = attention_loss
    self.cvae_loss = cvae_loss
    self.reconstruction_loss = reconstruction_loss
    self.total_loss = loss



  def define_train_op(self, config):
    train_op, optimizer = train_util.define_train_ops(
      self.total_loss, **config
    )
    self.train_op = train_op


  def add_summaries(self):
    """Adds model summaries."""

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


class AttentionModel(Model):
  """Test the attention network"""

  def build_model(self, config):
    self.attention = networks.AttentionNetwork(**config)

    # provide a fake datasource for masks
    batch_size, h, w, c = self.real_images.shape
    next_scope = tf.zeros((batch_size, h, w, 1))

    scopes = [next_scope]
    masks = []
    for i in range(config['attention_steps'] - 1):
      mask, next_scope = self.attention(self.real_images, next_scope)
      scopes.append(next_scope)
      masks.append(mask)
    masks.append(next_scope)

    self.masks = masks
    self.scopes = scopes


  def define_train_op(self, config):
    """
    Should be much easier to train! (they're essentially independent)

    For this second part, we're going to make the LAST THREE masks count
    Let's see if this trains better
    """
    c = self.real_images.get_shape().as_list()[-1]
    masks = tf.exp(tf.concat(self.masks[1:], axis=c))
    reconstructed_image = tf.minimum(c * masks, 1.0)
    self.total_loss = tf.losses.mean_squared_error(self.real_images, reconstructed_image)
    self.train_op, _ = train_util.define_train_ops(self.total_loss, **config)
    self.reconstructed_image = reconstructed_image


  def add_summaries(self):
    def _log_many(name, images):
      for i, m in enumerate(images):
        tf.summary.image(f'{name}/step{i}', m[0:1])
        tf.summary.histogram(f'{name}/step{i}', m[0])

    _log_many('mask', [tf.exp(x) for x in self.masks])
    _log_many('scope', [tf.exp(x) for x in self.scopes])
    
    tf.summary.histogram('input', self.real_images[0])

    tf.summary.image('inferred_image', self.reconstructed_image)
    tf.summary.image('input images', self.real_images)
    tf.summary.scalar('loss', self.total_loss)

registry = {
  'monet': MonetModel,
  'attention': AttentionModel,
  'vae': VAEModel
}
