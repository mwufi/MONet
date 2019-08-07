"""Generate 2M examples of different shapes, colors
"""

import pygame
import math
import random
from absl import logging
import time
import numpy as np
import tensorflow as tf

import sys, os
sys.path.append(os.getcwd())
from configs.datagen import hparams
from lib.util import expand_path


pygame.init()
screen = pygame.display.set_mode((128, 128))
done = False
clock = pygame.time.Clock()

# -------------- Utility methods ----------------
def make_star(x, y, points, radius, shrink=True):
  num_points = points
  point_list = []
  counter = 0 # you can make it rotate!

  n_points = num_points * 2 if shrink else num_points
  for i in range(n_points):
    r = radius
    if shrink and i % 2 == 0:
      r = radius // 2

    ang = i * 3.14159 / num_points
    if not shrink:
      ang *= 2
    ang += counter * 3.14159 / 60
    point_x = x + int(math.cos(ang) * r)
    point_y = y + int(math.sin(ang) * r)
    point_list.append((point_x, point_y))
  return point_list

def random_color():
  color = pygame.Color(100, 20, 251)
  color.hsla = (random.randint(0, 359), *color.hsla[1:])
  return color

def increment(c):
  hsla = [(c.hsla[0] + 4) % 360]  + list(c.hsla[1:])
  c.hsla = hsla
  return c

colors = {
  'purple': pygame.Color(100, 20, 251)
}

class Painter:
  def __init__(self, screen):
    self.screen = screen

  def draw(self, objects):
    for i in objects:
      self._draw(i)

  def _draw(self, attributes):
    name, pos, color, size = attributes
    x,y = pos
    if isinstance(color, str):
      color = colors[color]

    screen = self.screen
    if name == 'blob':
      pygame.draw.rect(screen, color, pygame.Rect(x, y, size, size))
      pygame.draw.circle(screen, color, (x, y), size)
    elif name == 'star':
      pygame.draw.polygon(screen, color, make_star(x, y, points=12, radius=size))
    elif name == 'circle':
      pygame.draw.circle(screen, color, (x, y), size)
    elif name == 'rect':
      pygame.draw.rect(screen, color, pygame.Rect(x, y, size*1.5, size*1.5))
    elif name == 'triangle':
      pygame.draw.polygon(screen, color, make_star(x, y, points=3, radius=size))
    elif name == 'diamond':
      pygame.draw.polygon(screen, color, make_star(x, y, points=4, radius=size, shrink=False))


def make_random_object(**kwargs):
  shape = random.choice(kwargs['shapes'])
  random_location = (random.randint(*kwargs['x_range']), random.randint(*kwargs['y_range']))
  obj = [shape, random_location, random_color(), kwargs['size']]
  return obj

def randomScene(**kwargs):
  return [make_random_object(**kwargs) for i in range(kwargs['n_objects'])]


def default_specs():
  return {
    'n_objects': 5,
    'shapes': ['rect', 'diamond', 'circle', 'triangle'],
    'size': 10,
    'x_range': [10, 110],
    'y_range': [10, 110]
  }


class RecordsWriter:
  def __init__(self, name):
    path = expand_path(name)
    if not os.path.exists(path):
      os.mkdir(path)
    self.filename = os.path.join(path, 'record_')

  def save(self, image, **kwargs):
    """Saves a scene"""
    raise NotImplementedError

  def close(self):
    pass


class NumpyBuffer(RecordsWriter):
  """Writes a bunch of npy files"""
  def __init__(self, name, write_every=1024):
    super(NumpyBuffer, self).__init__(name=name)
    self.images = []
    self.length = 0
    self.write_every = write_every
    self.written_index = 0

  def save(self, image, **kwargs):
    """Append to an array which flushes"""
    if self.length == self.write_every:
      self._flush()
    else:
      self.images.append(image)
      self.length += 1

  def _flush(self):
    np.save(self.filename + str(self.written_index), np.stack(self.images))
    self.written_index += 1
    self.images = []
    self.length = 0

  def close(self):
    self._flush()


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFRecordsWriter(RecordsWriter):
  """Writes a single .tfrecord file"""
  def __init__(self, name, **kwargs):
    super(TFRecordsWriter, self).__init__(name)
    self.writer = tf.io.TFRecordWriter(self.filename + '1.tfrecord')

  def save(self, image, **kwargs):
    n_objects = kwargs['n_objects']
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'image': _bytes_feature(image.tobytes()),
        'n_objects': _int64_feature(n_objects)
      }))
    self.writer.write(example.SerializeToString())

  def close(self):
    self.writer.flush()
    self.writer.close()



b = Painter(screen)
f = TFRecordsWriter(hparams['output_dir'], write_every=10)

x = y = 30
color = random_color()

# Since we only have one variable (n_objects) changing across the data
# we'll generate the necessary proportions of each spec ahead of time
logging.set_verbosity(logging.INFO)
for i, proportion in enumerate(hparams['data_splits']['n_objects']):
  total_objects = int(proportion * hparams['n_scenes'])
  start = time.time()

  logging.info(f'Generating {total_objects} scenes with {i+1} objects...')
  for x in range(total_objects):
    scene_spec = default_specs()
    scene_spec['n_objects'] = i + 1
    objects = randomScene(**scene_spec)

    screen.fill((0, 0, 0))
    b.draw(objects)
    imgdata = pygame.surfarray.array3d(screen)
    f.save(imgdata, **scene_spec)
    pygame.display.flip()

    if x % 1e5 == 0 and x > 0:
      end = time.time() - start
      logging.info(f'{x}/{total_objects}\tTook {end} seconds')
  end = time.time() - start
  logging.info(f'{total_objects}/{total_objects}\tTook {end} seconds')

# Write any leftovers to file!
f.close()
