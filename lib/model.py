
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
  """Base class

  The point of the model is to wrap a network with some data,
  a loss function, an op to train it, and possible methods
  for working with the network. For example, in a generative model
  you might have some `generate()` methods.

  To define a new model, just implement `define_train_op` to return a
  new train op. Happy subclassing!
  """

  def __init__(self, batch_size, config):
    """Sets up a model with a `train_op`
    """
    data_helper = data_helpers.registry[config['data_type']](config)
    real_images = data_helper.provide_data(batch_size)

    self.config = config
    self.batch_size = batch_size
    self.data_helper = data_helper
    self.real_images = data_helper.provide_data(batch_size)
    self.build_model(config)

  def build_model(self, config):
    """
    Usually:
    1) Define the model that takes `real_images` as input
    2) Define loss
    3) Define optimizer + train_op
    """
    raise NotImplementedError

  def add_summaries(self):
    """Logs the input data by default"""
    tf.summary.histogram('input', self.real_images[0])
    tf.summary.image('input images', self.real_images)


class VAEModel(Model):
  """The component VAE by itself"""

  def build_model(self, config):
    # our VAE takes as input: log attention masks
    _, h, w, c = self.real_images.get_shape().as_list()
    log_attention = tf.zeros((self.batch_size, h, w, 1))

    # and produces unnormalized versions of the inputs
    object_logits, mask_logits, latents = networks.ComponentVAE(**config)(
      tf.concat([self.real_images, log_attention], axis=3)
    )

    # In order to make the mask_logits form valid masks, we will
    # do a softmax over all the K attention steps. In this case, there
    # is only 1 step.
    mask = tf.sigmoid(mask_logits)

    # Let's look at the reconstructed images! This is not necessary to compute
    # the reconstruction loss (as we'll see later)
    obj_image = tf.sigmoid(object_logits)
    masked_image = tf.multiply(mask, obj_image)

    ##### So how DO we compute the reconstruction loss?
    # Gaussian decoder distribution p_(x| z_k)
    variance = config['component_scale_background']
    log_likelihood = (-0.5 * tf.log(variance) - tf.square(obj_image - self.real_images) / (2 * variance))
    rec_loss = -tf.reduce_logsumexp(log_attention + log_likelihood, axis=[1,2,3])
    reconstruction_loss = tf.reduce_sum(rec_loss)

    # Part 2: KL divergence between the latents and our prior N(0,1)
    mu, logvar = tf.split(latents, num_or_size_splits=2, axis=1)
    KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))

    self.endpoints = {
      'raw_image': object_logits,
      'raw_mask': mask_logits,
      'image': obj_image,
      'mask': mask,
      'masked_image': masked_image,
      'latents': latents,
      'reconstruction_loss': reconstruction_loss,
      'KL_loss': KLD
    }

    self.total_loss = reconstruction_loss + KLD
    self.train_op, optimizer = train_util.define_train_ops(self.total_loss, **config)

  def add_summaries(self):
    super(VAEModel, self).add_summaries()
    scalars = ['reconstruction_loss', 'KL_loss']
    images = ['image', 'mask', 'masked_image']
    histograms = ['latents']

    for i in histograms:
      tf.summary.histogram(i, self.endpoints[i]) 
    for i in scalars:
      tf.summary.scalar(i, self.endpoints[i])
    for i in images:
      tf.summary.image(i, self.endpoints[i])


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

    self.train_op, optimizer = train_util.define_train_ops(
      self.total_loss, **config
    )


  def add_summaries(self):
    """Adds model summaries."""
    super(MonetModel, self).add_summaries()

    def _log_many(name, images):
      for i, m in enumerate(images):
        tf.summary.image(f'step{i}/{name}', m)
        tf.summary.histogram(f'step{i}/{name}', m[0])

    _log_many('attention_mask', self.attn_masks)
    _log_many('cvae_mask', self.obj_masks)
    _log_many('cvae_image', self.obj_images)

    tf.summary.image('inferred_images', self.reconstructed_images)

    tf.summary.scalar('attention_KL', self.attention_loss)
    tf.summary.scalar('normal_vae_KL', self.cvae_loss)
    tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
    tf.summary.scalar('total_loss', self.total_loss)


class AttentionModel(Model):
  """Test the attention network"""

  def build_model(self, config):
    self.attention = networks.AttentionNetwork(**config)

    ##### provide a fake datasource for masks
    batch_size, h, w, c = self.real_images.get_shape().as_list()
    next_scope = tf.zeros((batch_size, h, w, 1))

    ##### Forward pass
    scopes = [next_scope]
    masks = []
    for i in range(config['attention_steps'] - 1):
      mask, next_scope = self.attention(self.real_images, next_scope)
      scopes.append(next_scope)
      masks.append(mask)
    masks.append(next_scope)

    masks_tensor = tf.exp(tf.concat(masks[1:], axis=c))
    reconstructed_image = tf.minimum(c * masks_tensor, 1.0)

    ##### Add variables as properties
    self.masks = masks
    self.scopes = scopes
    self.reconstructed_image = reconstructed_image
    self.total_loss = tf.losses.mean_squared_error(self.real_images, reconstructed_image)

    self.train_op, optimizer = train_util.define_train_ops(
      self.total_loss, **config
    )


  def add_summaries(self):
    def _log_many(name, images):
      for i, m in enumerate(images):
        tf.summary.image(f'{name}/step{i}', m[0:1])
        tf.summary.histogram(f'{name}/step{i}', m[0])

    _log_many('mask', [tf.exp(x) for x in self.masks])
    _log_many('scope', [tf.exp(x) for x in self.scopes])
    tf.summary.image('inferred_image', self.reconstructed_image)
    tf.summary.scalar('loss', self.total_loss)

registry = {
  'monet': MonetModel,
  'attention': AttentionModel,
  'vae': VAEModel
}
