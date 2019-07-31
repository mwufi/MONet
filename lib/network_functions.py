
"""Builds the vae and attention network functions"""

from lib.networks import component_vae, attention_network


def component_vae(inputs, **kwargs):
  """Builds a vae that outputs masks and components"""
  return component_vae(inputs, **kwargs)

g_fn_registry = {
    'component_vae': component_vae,
}