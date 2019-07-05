# author: Mengmeng Zhu, Purdue University
# email: zhu457@purdue.edu
# Nov 16, 2017

import tensorflow as tf
import numpy as np
import glob

def generate_filename_queue(data_dir, data_set='train'):
  filenames = glob.glob(data_dir + data_set + '*.tfrecords')
  filename_queue = tf.train.string_input_producer(filenames)
  return filename_queue

def read_and_decode(filename_queue, image_shape=[25, 25, 25, 2]):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.decode_raw(features['image_raw'], tf.float32)
  image.set_shape([np.prod(image_shape)])
  image = tf.reshape(image, image_shape)

  label = tf.cast(features['label'], tf.int32)

  return image, label

def generate_batch(image, label, batch_size, capacity, min_after_dequeue, 
  num_threads):
  images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size, capacity=capacity,
    min_after_dequeue=min_after_dequeue, num_threads=num_threads)
  return images_batch, labels_batch

def queue(data_dir, data_set='train', image_shape=[25, 25, 25, 2], 
  batch_size=128, num_examples=None, num_threads=16):
  filename_queue = generate_filename_queue(data_dir, data_set)
  image, label = read_and_decode(filename_queue, image_shape=image_shape)
  if num_examples is None:
    capacity = 10000
    min_after_dequeue = 8000
  else:
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples * min_fraction_of_examples_in_queue)
    capacity = min_queue_examples + 3 * batch_size
    min_after_dequeue = min_queue_examples
  images_batch, labels_batch = generate_batch(image, label, 
    batch_size, capacity, min_after_dequeue, num_threads)
  return images_batch, labels_batch

def get_train_batch(data_dir, image_shape=[25, 25, 25, 2], batch_size=128,  
  num_examples=None, num_threads=16):
  X_train_batch, y_train_batch = queue(data_dir, image_shape=image_shape, 
    batch_size=batch_size, num_examples=num_examples, num_threads=num_threads)
  return X_train_batch, y_train_batch

def get_val_batch(data_dir, image_shape=[25, 25, 25, 2], batch_size=128, 
  num_examples=None, num_threads=16):
  X_val_batch, y_val_batch = queue(data_dir, data_set='val', 
    image_shape=image_shape, batch_size=batch_size, num_examples=num_examples, 
    num_threads=num_threads)
  return X_val_batch, y_val_batch

