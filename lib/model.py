
from lib import network_functions as net_fns

def make_model(config):
  # Get network functions and wrap with hparams
  g_fn = lambda x: net_fns.g_fn_registry[config['g_fn']](x, **config)
  d_fn = lambda x: net_fns.d_fn_registry[config['d_fn']](x, **config)