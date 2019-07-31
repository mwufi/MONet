"""Config for data generation"""

# Define Config
hparams = {}
hparams['n_scenes'] = 2e3
hparams['data_splits'] = {
  # 3-6 objects in our scenes
  'n_objects': [0.0, 0.0, 0.25, 0.25, 0.25, 0.25]
}