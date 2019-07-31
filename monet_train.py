"""Train a MONet model.

See https://arxiv.org/abs/1901.11390 for details about the model.

"""

from lib import datasets
from configs import mini

d = datasets.BaseDataset(mini.hparams)