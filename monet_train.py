"""Train a MONet model.

See https://arxiv.org/abs/1901.11390 for details about the model.

"""

from lib import datasets, data_helpers
from configs import mini
import tensorflow as tf
from tensorflow.compat.v1 import Session, global_variables_initializer, tables_initializer

d = datasets.BaseDataset(mini.hparams)

e = data_helpers.DataImagesHelper(mini.hparams)
real_images = e.provide_data(64)

with Session() as sess:
  sess.run([global_variables_initializer(), tables_initializer()])
  images = sess.run(real_images)
  print('Successfully read batch!', images.shape)