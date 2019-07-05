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

data_dir = '../data_10classes/'
filename_queue = generate_filename_queue(data_dir)
image, label = read_and_decode(filename_queue)

init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
sess.run(init)

image_data, label_data = sess.run([image, label])
