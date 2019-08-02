"""Train a MONet model.

See https://arxiv.org/abs/1901.11390 for details about the model.

"""

from lib import datasets, data_helpers
from configs import  mini
from lib.networks import MONet
import time

import tensorflow as tf
from tensorflow.compat.v1 import Session, global_variables_initializer, tables_initializer

config = mini.hparams
d = datasets.BaseDataset(config)
e = data_helpers.DataImagesHelper(config)
real_images = e.provide_data(32)
monet = MONet(**config)
reconstructed_image, endpoints = monet(real_images)

with Session() as sess:
  sess.run([global_variables_initializer(), tables_initializer()])
  images = sess.run(real_images)
  print('Successfully read batch!', images.shape)

  start = time.time()
  image = sess.run(reconstructed_image)
  attention_masks = sess.run(endpoints['attention_mask'])
  print('Image', image.shape)
  print(f'{len(attention_masks)} attention masks')
  print(f'-- {time.time() - start:.2f} s')

  tf.summary.image('image', image, max_outputs=16)
  for i, mask in enumerate(attention_masks):
    tf.summary.image(f'attention_mask_{i}', mask, max_outputs=16)
  summary = sess.run(tf.summary.merge_all())

  file_writer = tf.summary.FileWriter('temp_graph', sess.graph)
  file_writer.add_summary(summary, global_step=0)

  file_writer.close()


