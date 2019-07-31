import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.layers import instance_norm
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D, MaxPool2D, Dense, Flatten


def down_block(x, filters, downsample=True, **kwargs):
  x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='truncated_normal', use_bias=False)(x)
  x = instance_norm(x)
  x = residual = ReLU()(x)
  if downsample:
    x = MaxPool2D()(residual)
  return x, residual


def up_block(x, residual, filters, upsample=True, **kwargs):
  x = tf.concat([x, residual], axis=3)
  x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='truncated_normal', use_bias=False)(x)
  x = instance_norm(x)
  x = ReLU()(x)
  if upsample:
    x = UpSampling2D(interpolation='nearest')(x)
  return x


def mlp(y, **kwargs):
  """This MLP returns a tensor in the original shape as the input"""
  h, w, c = y.shape[1:]
  y = Flatten()(y)
  for t in kwargs['mlp_sizes']:
    y = Dense(t, activation='relu')(y)
  y = Dense(h * w * c, activation='relu')(y)
  y = tf.reshape(y, (-1, h, w, c))
  return y


def unet(x, **kwargs):
  residuals = []
  for i in range(kwargs['num_blocks']):
    to_downsample = i < (kwargs['num_blocks'] - 1)
    x, residual = down_block(x, filters=kwargs['filters'][i], downsample=to_downsample)
    residuals.append(residual)

  # the non-residual block is a 3-layer MLP
  y = mlp(x, **kwargs)

  for i in range(kwargs['num_blocks']):
    to_upsample = i < (kwargs['num_blocks'] - 1)
    y = up_block(y, residuals[-(i + 1)],
                 filters=kwargs['filters'][-(i + 1)],
                 upsample=to_upsample)

  # Finally, you get a 1x1 channel
  y = Conv2D(1, 1, activation='relu', kernel_initializer='truncated_normal')(y)
  return y


def attention_network(image, log_attention_scope, iteration, **kwargs):
  x = tf.concat([image, log_attention_scope], axis=3)
  log_min = 1e-10
  log_max = 1e8

  if iteration == kwargs['attention_steps'] - 1:
    log_mask = log_attention_scope
    log_next_scope = None
  else:
    logits = unet(log_attention_scope, **kwargs)
    log_alpha_k = tf.math.log_sigmoid(logits)
    log_mask = log_attention_scope + log_alpha_k
    log_next_scope = log_attention_scope + tf.math.log(
      tf.clip_by_value((1 - tf.exp(log_alpha_k)), clip_value_min=log_min, clip_value_max=log_max))

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


class ComponentVAE(snt.AbstractModule):

  def __init__(self, name='component_vae', **kwargs):
    super().__init__(name=name)
    self.kwargs = kwargs
    self.encoder_args = self.kwargs['cvae_encoder']
    self.decoder_args = self.kwargs['cvae_decoder']
    self.latent_dim = self.encoder_args['mlp'][-1] // 2

    self.encoder_cnn = snt.nets.ConvNet2D(
      output_channels=self.encoder_args['output_channels'],
      kernel_shapes=self.encoder_args['kernel_shapes'],
      strides=self.encoder_args['strides'],
      paddings=[snt.SAME],
      # By default final layer activation is disabled.
      activate_final=True,
      name='convolutional_module'
    )
    self.encoder_mlp = snt.nets.MLP(
      output_sizes=self.encoder_args['mlp'],
      name='fully_connected_module'
    )
    self.decoder_cnn = snt.nets.ConvNet2D(
      output_channels=self.decoder_args['output_channels'],
      kernel_shapes=self.decoder_args['kernel_shapes'],
      strides=self.decoder_args['strides'],
      paddings=[snt.VALID],
      # By default final layer activation is disabled.
      activate_final=True,
      name='convolutional_module_too'
    )


  def encode(self, inputs):
    outputs = self.encoder_cnn(inputs)
    outputs = snt.BatchFlatten()(outputs)
    outputs = self.encoder_mlp(outputs)
    return outputs


  def sample(self, latents):
    bs = latents.shape[0]
    z_mu, z_log_sigma = tf.split(latents, num_or_size_splits=2, axis=1)
    z_samples = tf.random.normal(shape=(bs, self.latent_dim)) * tf.exp(z_log_sigma * .5) + z_mu
    return z_samples


  def decode(self, latents):
    decoder_inputs = broadcast_decoder(latents, **self.kwargs)
    outputs = self.decoder_cnn(decoder_inputs)
    return outputs


  def _build(self, inputs):
    """ Build the model stack.
    """
    latents = self.encode(inputs)
    z = self.sample(latents)
    outputs = self.decode(z)
    rgb_image, reconstructed_mask = outputs[:,:,:,:3], outputs[:,:,:,3]
    return rgb_image, reconstructed_mask


class MONet(snt.AbstractModule):

  def __init__(self, name='monet', **kwargs):
    super().__init__(name=name)

    self.kwargs = kwargs
    self.attention_steps = self.kwargs['attention_steps']
    self.cvae = ComponentVAE(**kwargs)

  def _build(self, image, **kwargs):
    batch_size, h, w, c = image.shape
    scope = tf.zeros((batch_size, h, w, 1))

    endpoints = {}
    endpoints['attention_mask'] = []
    endpoints['reconstruct_mask'] = []
    endpoints['reconstruct_image'] = []

    reconstructed_image = tf.zeros_like(image)
    for i in range(self.attention_steps):
      # feed it to our attention network
      log_mask, scope = attention_network(image, scope, iteration=i, **self.kwargs)
      attention_mask = tf.exp(log_mask)
      endpoints['attention_mask'].append(attention_mask)

      # feed it to the VAE
      object_mean, mask_logits = self.cvae(tf.concat([image, log_mask], axis=3))
      noise_scale = hparams['component_scale_background'] if i == 0 else hparams['component_scale_foreground']
      object_image = tf.random.normal(object_mean.shape) * noise_scale + object_mean
      reconstructed_mask = tf.sigmoid(mask_logits)
      endpoints['reconstruct_mask'].append(reconstructed_mask)
      endpoints['reconstruct_image'].append(object_image)
      reconstructed_image += object_image

    return reconstructed_image, endpoints


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
    log_mask, scope = attention_network(image, scope, iteration=i, **hparams)
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


def test_monet(image, scope, hparams):
  m = MONet(**hparams)
  reconstructed_image, endpoints = m(image)

  print('Reconstructed image', image.shape)
  plt.imshow(tf.squeeze(reconstructed_image[0]))
  plt.title('Reconstructed image')
  plt.show()

  check_attention_masks(endpoints['attention_mask'])


if __name__ == '__main__':
  import numpy as np
  import sys, os

  sys.path.append(os.getcwd())
  import matplotlib.pyplot as plt
  from configs.mini import hparams

  tf.enable_eager_execution()

  # make an image + scope
  batch_size = 2
  h, w = hparams['img_size']
  image = tf.random.uniform((batch_size, h, w, 3))
  sd = np.zeros((batch_size, h, w, 3))
  sd[0, 0:10, 0:10, 0] = 0
  scope = tf.convert_to_tensor(sd, dtype=tf.float32)

  test_monet(image, scope, hparams)
