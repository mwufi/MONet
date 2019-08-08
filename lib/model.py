
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
    #### Monet needs... images!
    monet = networks.MONet(**config)
    endpoints = monet(self.real_images)

    #### Let's extract all the endpoints (except for scope)
    log_attn = endpoints['mask']
    object_logits = endpoints['raw_image']
    object_images = [tf.sigmoid(o) for o in object_logits]
    mask_logits = endpoints['raw_mask']
    latents = endpoints['latents']

    #### Define the loss
    with tf.name_scope('loss'):
      # Part 1. Gaussian decoder distribution p_(x| z_k) - with MASK
      reconstruction_loss = 0
      for log_attention, image in zip(log_attn, object_images):
        variance = config['component_scale_background']
        log_likelihood = (-0.5 * tf.log(variance) - tf.square(image - self.real_images) / (2 * variance))
        rec_loss = -tf.reduce_logsumexp(log_attention + log_likelihood, axis=[1,2,3])
        reconstruction_loss += tf.reduce_sum(rec_loss)
      # Part 2: KL divergence between the latents and our prior N(0,1)
      KLD = 0
      for t in latents:
        mu, logvar = tf.split(t, num_or_size_splits=2, axis=1)
        KLD += -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
      # Part 3: KL divergence between the attention masks and reconstructed mask
      # in order for the vae to return valid masks, we need to sigmoid it
      mask_reconstruction = 100 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.exp(tf.concat(log_attn, axis=3)),
          logits=tf.concat(mask_logits, axis=3)))
      # Part 4: Total loss is a weighted sum
      self.total_loss = reconstruction_loss + config['beta'] * KLD + config['gamma'] * mask_reconstruction

    #### Most of these endpoints are LISTS - corresponding to one attention step
    masks = [tf.nn.sigmoid(x) for x in mask_logits]
    attention_masks = [tf.exp(x) for x in log_attn]
    masked_images = [tf.multiply(m, i) for m, i in zip(attention_masks, object_images)]
    self.endpoints = {
      'attention_masks': attention_masks,
      'attention_scopes': endpoints['scope'],
      'mask_images': masks,
      'object_images': object_images,
      'masked_images': masked_images,
      'latents': latents,
      'reconstruction_loss': reconstruction_loss,
      'KLD': KLD,
      'mask_reconstruction': mask_reconstruction,
    }

    #### Finally, define the train op!
    self.train_op, optimizer = train_util.define_train_ops(self.total_loss, **config)

  def add_summaries(self):
    """Adds model summaries."""
    super(MonetModel, self).add_summaries()

    def _log_many(name, images):
      """Display images of the same type together"""
      for i, m in enumerate(images):
        tf.summary.image(f'{name}/step{i}', m[0:1])
        tf.summary.histogram(f'{name}/step{i}', m[0])

    for t in [
      'attention_masks',
      'attention_scopes',
      'mask_images',
      'object_images',
      'masked_images'
    ]:
      _log_many(t, self.endpoints[t])

    for t in [
      'reconstruction_loss',
      'KLD',
      'mask_reconstruction'
    ]:
      tf.summary.scalar(t, self.endpoints[t])

    for t in [
      'latents'
    ]:
      tf.summary.histogram(t, self.endpoints[t])

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
