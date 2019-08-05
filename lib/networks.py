import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.layers import instance_norm
from tensorflow.keras.layers import ReLU, UpSampling2D, MaxPool2D


class DownBlock(snt.AbstractModule):
  def __init__(self, name='down_block', **kwargs):
    super(DownBlock, self).__init__(name=name)
    self._downsample = kwargs['downsample']
    self._filters = kwargs['filters']

  def _build(self, x):
    x = snt.Conv2D(self._filters, 3, padding='SAME', use_bias=False)(x)
    x = instance_norm(x)
    x = residual = ReLU()(x)
    if self._downsample:
      x = MaxPool2D()(residual)
    return x, residual


class UpBlock(snt.AbstractModule):
  def __init__(self, name='up_block', **kwargs):
    super(UpBlock, self).__init__(name=name)
    self._upsample = kwargs['upsample']
    self._filters = kwargs['filters']

  def _build(self, x, residual):
    x = tf.concat([x, residual], axis=3)
    x = snt.Conv2D(self._filters, 3, padding='SAME', use_bias=False)(x) # default initializer is a truncated normal! how handy
    x = instance_norm(x)
    x = ReLU()(x)
    if self._upsample:
      x = UpSampling2D(interpolation='bilinear')(x)
    return x


class UNet(snt.AbstractModule):
  def __init__(self, name='u_net', **kwargs):
    super(UNet, self).__init__(name=name)
    self._numBlocks = kwargs['num_blocks']
    self._filters = kwargs['filters']
    self._mlp = kwargs['mlp_sizes']

  def _build(self, x):
    residuals = []
    for i in range(self._numBlocks):
      to_downsample = i < (self._numBlocks - 1)
      x, residual = DownBlock(filters=self._filters[i], downsample=to_downsample)(x)
      residuals.append(residual)

    # the non-residual block is a 3-layer MLP
    h, w, c = x.get_shape().as_list()[1:]
    y = snt.BatchFlatten()(x)
    y = snt.nets.MLP(
      output_sizes=self._mlp + [h*w*c],
      name='mlp'
    )(y)
    y = snt.BatchReshape(shape=(h, w, c))(y)

    for i in range(self._numBlocks):
      to_upsample = i < (self._numBlocks - 1)
      y = UpBlock(filters=self._filters[-(i + 1)],
                   upsample=to_upsample)(y, residuals[-(i + 1)])

    # Finally, you get a 1x1 channel
    y = snt.Conv2D(1, 1)(y)
    return y


class AttentionNetwork(snt.AbstractModule):
  def __init__(self, name='attention_network', **kwargs):
    super(AttentionNetwork, self).__init__(name=name)
    self._kwargs = kwargs

  def _build(self, image, log_attention_scope):
    x = tf.concat([image, log_attention_scope], axis=3)
    log_min = 1e-10
    log_max = 1e8

    logits = UNet(**self._kwargs)(log_attention_scope)
    # log(\alpha_k) and log(1-\alpha_k)
    log_alpha_k = tf.math.log_sigmoid(logits)
    log_beta_k = tf.math.log_sigmoid(tf.ones_like(logits)-logits)
    log_mask = log_attention_scope + log_alpha_k
    log_next_scope = log_attention_scope + log_beta_k

    return log_mask, log_next_scope


def expand(tensor, *dims):
  """Expand multiple axes, sequentially"""
  for i in dims:
    tensor = tf.expand_dims(tensor, axis=i)
  return tensor


def broadcast_decoder(z, **kwargs):
  """Make inputs for the broadcast decoder

  Inputs: (BATCH_SIZE, LATENT_DIM) vector z

  Output: tensor w/ shape (BATCH_SIZE, HEIGHT, WIDTH, LATENT_DIM + 2)
  """
  h, w = kwargs['img_size']
  h += 8  # we add 8 because the output CNN has no padding :)
  w += 8
  batch = z.shape[0]

  x = tf.lin_space(-1.0, 1.0, h)
  y = tf.lin_space(-1.0, 1.0, w)
  x, y = tf.meshgrid(x, y)
  x, y = expand(x, 0, -1), expand(y, 0, -1)
  x, y = tf.tile(x, [batch, 1, 1, 1]), tf.tile(y, [batch, 1, 1, 1])

  z = expand(z, 1, 1)
  zs = tf.tile(z, (1, h, w, 1))
  return tf.concat([x, y, zs], axis=-1)


def sample(latents):
  z_mu, z_log_variance = tf.split(latents, num_or_size_splits=2, axis=1)
  z_samples = tf.random.normal(shape=z_mu.shape) * tf.exp(z_log_variance * .5) + z_mu
  return z_samples


class ComponentVAE(snt.AbstractModule):

  def __init__(self, name='component_vae', **kwargs):
    super().__init__(name=name)
    self.kwargs = kwargs
    self.encoder_args = self.kwargs['cvae_encoder']
    self.decoder_args = self.kwargs['cvae_decoder']
    self.latent_dim = self.encoder_args['mlp'][-1] // 2

  @snt.reuse_variables
  def encode(self, inputs):
    """Turns inputs into Gaussian posterior mean+log variance"""
    outputs = snt.nets.ConvNet2D(
      output_channels=self.encoder_args['output_channels'],
      kernel_shapes=self.encoder_args['kernel_shapes'],
      strides=self.encoder_args['strides'],
      paddings=[snt.SAME],
      # By default final layer activation is disabled.
      activate_final=True,
      name='encoder'
    )(inputs)
    outputs = snt.BatchFlatten()(outputs)
    outputs = snt.nets.MLP(
      output_sizes=self.encoder_args['mlp'],
      name='fully_connected_module'
    )(outputs)
    return outputs


  @snt.reuse_variables
  def decode(self, latents):
    """Outputs logits (ie, [-inf, inf]) for the mask and object image
    """
    decoder_inputs = broadcast_decoder(latents, **self.kwargs)
    outputs = snt.nets.ConvNet2D(
      output_channels=self.decoder_args['output_channels'],
      kernel_shapes=self.decoder_args['kernel_shapes'],
      strides=self.decoder_args['strides'],
      paddings=[snt.VALID],
      name='decoder'
    )(decoder_inputs)
    return outputs

  def _build(self, inputs):
    """ Build the model stack.
    """
    latents = self.encode(inputs)
    z = sample(latents)
    outputs = self.decode(z)
    rgb_image, reconstructed_mask = outputs[:,:,:,:3], outputs[:,:,:,3:]
    return rgb_image, reconstructed_mask, latents


class MONet(snt.AbstractModule):
  """A Multi-Object Network"""

  def __init__(self, name='monet', **kwargs):
    super().__init__(name=name)
    self._kwargs = kwargs
    self._attention_steps = self._kwargs['attention_steps']
    with self._enter_variable_scope():
      self.attention = AttentionNetwork(**self._kwargs)
      self.cvae = ComponentVAE(**self._kwargs)

  def _build(self, image):
    """Infers the objects in the image

    Returns:
      reconstructed_image: the image from putting all the inferred objects together
      endpoints:
        attention_mask: shape (B,H,W, attention_step) tensor: attention masks returned by the attention network.
        log_obj_mask: shape (B,H,W, attention_step) tensor: log of the masks returned by the Component VAE.
        obj_image: a list of the masked images by the Component VAE
      """
    print('OK, building the thing')
    batch_size, h, w, c = image.shape
    log_attention_scope = tf.zeros((batch_size, h, w, 1))

    endpoints = {}
    endpoints['attention_mask'] = []
    endpoints['log_attention_mask'] = []
    endpoints['mask_logits'] = []
    endpoints['obj_mask'] = []
    endpoints['obj_image'] = []
    endpoints['obj_latent'] = []

    for i in range(self._attention_steps):
      if i < self._attention_steps - 1:
        print('Attention step', i)
        log_mask, log_attention_scope = self.attention(image, log_attention_scope)
      else:
        log_mask = log_attention_scope
      attention_mask = tf.exp(log_mask, name='mask_attention')
      endpoints['attention_mask'].append(attention_mask)
      endpoints['log_attention_mask'].append(log_mask)

      # feed it to the VAE
      object_logits, mask_logits, latents = self.cvae(tf.concat([image, log_mask], axis=3))
      endpoints['mask_logits'].append(mask_logits)
      endpoints['obj_mask'].append(tf.nn.sigmoid(mask_logits))
      endpoints['obj_image'].append(tf.nn.sigmoid(object_logits))
      endpoints['obj_latent'].append(latents)

    return endpoints


def check_attention_masks(masks):
  """See if the list of masks sum to 1"""
  masks = tf.stack(masks, axis=0)
  sum_masks = tf.reduce_sum(masks, axis=0)
  if np.allclose(sum_masks, np.ones_like(sum_masks)):
    print("Masks sum to 1!")
  else:
    print(sum_masks)


def test_separate(image, scope, hparams):
  test_attention_network = False
  test_vae = True
  test_image = True

  # initialize networks
  myLittlePony = ComponentVAE(**hparams)

  # feed it to our attention network
  masks = []
  for i in range(hparams['attention_steps']):
    log_mask, scope = AttentionNetwork(hparams)(image, scope, iteration=i)
    masks.append(tf.exp(log_mask))

    # feed it to the VAE
    image, mask_logits = myLittlePony(tf.concat([image, log_mask], axis=3))
    noise_scale = hparams['component_scale_background'] if i == 0 else hparams['component_scale_foreground']
    image = tf.random.normal(image.shape) * noise_scale + image
    mask = tf.sigmoid(mask_logits)

    if test_attention_network:
      print(log_mask.shape, scope.shape if scope is not None else 'Done')
      if scope is not None:
        print('Log Mask')
        print(log_mask[0, 5:15, 0:10, 0])
        plt.imshow(tf.squeeze(tf.exp(log_mask[0])))
        plt.colorbar()
        plt.title('Attention mask')
        plt.show()
    if test_vae:
      print('Component VAE', mask.shape)
      plt.imshow(tf.squeeze(mask[0]))
      plt.title('Reconstruct4ed Mask')
      plt.show()
    if test_image:
      print('Image', image.shape)
      print(image[0])
      plt.imshow(tf.squeeze(image[0]))
      plt.title('Image component')
      plt.show()

  check_attention_masks(masks)


def test_monet(image, scope, eager=False, **hparams):
  m = MONet(**hparams)
  reconstructed_image, endpoints = m(image)

  if eager:
    print('Reconstructed image', image.shape)
    plt.imshow(tf.squeeze(reconstructed_image[0]))
    plt.title('Reconstructed image')
    plt.show()

    check_attention_masks(endpoints['attention_mask'])
  else:
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter("temp_graph", sess.graph)
      writer.close()


if __name__ == '__main__':
  import numpy as np
  import sys, os

  sys.path.append(os.getcwd())
  import matplotlib.pyplot as plt
  from configs.mini import hparams

  # tf.enable_eager_execution()

  # make an image + scope
  batch_size = 2
  h, w = hparams['img_size']
  image = tf.random.uniform((batch_size, h, w, 3))
  sd = np.zeros((batch_size, h, w, 3))
  sd[0, 0:10, 0:10, 0] = 0
  scope = tf.convert_to_tensor(sd, dtype=tf.float32)

  test_monet(image, scope, eager=False, **hparams)