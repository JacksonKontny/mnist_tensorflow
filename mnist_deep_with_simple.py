"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.training import SummaryWriter

import tensorflow as tf

FLAGS = None



def conv2d(x, W):
  """
  conv2d returns a 2d convolution layer with full stride.

  This is a helper function for creating a convolutional layer, since we have
  to do that a lot.  Strides are just the distance that the convolutional
  window moves each set, while padding designates the padding algorithm to use.

  """
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """
  max_pool_2x2 downsamples a feature map by 2X.

  Similar to conv2d, this is a helper function to create a 2X2 pooling window
  """
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """
  weight_variable generates a weight variable of a given shape.

  This is a helper function to create the weight variable.  The dimension of
  the weight variable is passed in, and a random array of weights with a stdev
  of .1 is generated.
  """
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """
  bias_variable generates a bias variable of a given shape.

  This is a helper function to generate the biases.  Just like the weights,
  a vector the size of the input shape is created with random variables with
  stdev of 0.1
  """
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def simple_nn(x):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  W = tf.Variable(tf.zeros([784,10]))
  b = tf.Variable(tf.zeros([10]))

  y = tf.matmul(x, W) + b

  return y


def simple_main(_):

  # Read in the dataset
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create a placeholder for the input, which should be an arbitrary number of
  # 28 X 28 images which are flattened to a 1 X 784 vector
  x = tf.placeholder(tf.float32, [None, 784])

  # Set a placeholder for the labeled data - an arbitrary number of vectors of
  # length 10, 1 value for every possible output (digits from 0 to 9).
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Create the network variable 'y', which is the output of the simple neural
  # network implementation
  y = simple_nn(x)

  # Define the cross entropy loss of the network.  The cross entropy loss will
  # be the average cross entropy loss accross all samples
  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    )
  tf.summary.scaler('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    # Add in the gradient descent calculations into the simple neural network
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  with tf.name_scope('evaluation'):
    with tf.name_scope('correct'):
      # A correct prediction occurs if the network assigns the largest
      # probability to the label that is the correct label
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):

      # set up the accuracy computation - the sum of the correct predictions
      # over the size of the dataset
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scaler('accuracy', accuracy)
  merged = tf.summary.merge_all()
  train_writer = SummaryWriter(FLAGS.summaries_dir + '/train')
  test_writer = SummaryWriter(FLAGS.summaries_dir + '/test')

  # Initialize the session - this is where the computation happens
  with tf.Session() as sess:
    # Initialize the weights and biases
    sess.run(tf.initialize_all_variables())

    # Loop 1000 epochs
    for i in xrange(10001):
      # Get the next 100 examples in the batch
      batch = mnist.train.next_batch(100)

      # Run one epoch through the neural net with backpropagation.  Pass the
      # images and their labels as input for the network to make predictions
      # and evaluate the loss
      summary, _ = sess.run([merged, train_step],
                            feed_dict={x: batch[0], y_: batch[1]})
      train_writer.add_summary(summary)

      # for every 100 epocs...
      if i % 100 == 0:

        # Now evaluate what the accuracy is, using all images and labels
        # in the dataset
        summary, acc = sess.run(
          [merged, accuracy],
          feed_dict={x: mnist.test.images, y_: mnist.test.labels}
        )
        test_writer.add_summary(summary, acc)
        # print the accuracy for the given step
        print('step %d, test accuracy %g' % (i, acc))


def create_variable_summaries(var_list):
  """
  Attach a lot of summaries to a Tensor (for TensorBoard visualization).

  creates mean, standard deviation, min, max, and histogram summary info
  for the variable passed in
  """
  with tf.name_scope('summaries'):
    for var in var_list:
      # Get the mean value of the vector passed in
      mean = tf.reduce_mean(var)

      # tell tensorflow that this information will be stored for later display in
      # tensorboard
      tf.summary.scalar('mean', mean)

      # Get the standard deviation of the variable passed in
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

      # tell tensorflow that this information will be stored for later display in
      # tensorboard
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=simple_main)# , argv=[sys.argv[0]] + unparsed)
