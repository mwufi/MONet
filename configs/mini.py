
"""Config for training."""

# Hyperparameters
hifreqres = True
data_type = 'mel'  # 'linear', 'phase'
train_progressive = True
lr = 8e-4

# Define Config
hparams = {}
hparams['train_data_path'] = 'data2'
hparams['dataset_name'] = 'shapes'

# Attention network
hparams['num_blocks'] = 5
hparams['filters'] = [32,32,4,4,4]
hparams['mlp_sizes'] = [128, 128]
hparams['attention_steps'] = 4

# Component VAE
hparams['cvae_encoder'] = {
  'output_channels': [32, 32, 64, 64],
  'kernel_shapes': [3],
  'strides': [2],
  'mlp': [256, 32]
}
hparams['cvae_decoder'] = {
  'output_channels': [32, 32, 32, 32, 4],
  'kernel_shapes': [3, 3, 3, 3, 1],
  'strides': [1]
}

# 0.05 can be used for all component scales in the component VAE experiments in section 2.2
hparams['component_scale_background'] = 0.09
hparams['component_scale_foreground'] = 0.11
hparams['beta'] = 0.5
hparams['gamma'] = 0.25

hparams['img_size'] = (128, 128)

# optimizer
hparams['optimizer'] = 'RMSProp'
hparams['lr'] = 1e-4
hparams['batch_size'] = 64