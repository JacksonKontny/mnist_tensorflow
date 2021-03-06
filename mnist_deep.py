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


def create_variable_summaries(var_list):
  """
  Attach a lot of summaries to a Tensor (for TensorBoard visualization).

  creates mean, standard deviation, min, max, and histogram summary info
  for the variable passed in
  """
  with tf.name_scope('summaries'):
    for var_name, var in var_list:
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
      tf.summary.scalar('stddev_{}'.format(var_name), stddev)
      tf.summary.scalar('max_{}'.format(var_name), tf.reduce_max(var))
      tf.summary.scalar('min_{}'.format(var_name), tf.reduce_min(var))
      tf.summary.histogram('histogram_{}'.format(var_name), var)


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  # the -1 field specifies that we want the reshaped vector to 'fit' into the
  # existing vector that is getting reshaped.  Since the image is a flat
  # vector of 784 pixels, we will get a vector of dimensions
  # [N_examples, 28, 28, 1]
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  # This layer uses a 5 X 5 convolution.
  # the 'name_scope' param is used later when graphing the network visually.
  with tf.name_scope('conv1'):
    # The weight for the first layer should be a 5 X 5 convolution mapping with
    # one input channel and 32 output channels
    W_conv1 = weight_variable([5, 5, 1, 32])

    # The bias should just be the number of output channels
    b_conv1 = bias_variable([32])

    # compute the relu activation for the first convolutional layer
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    create_variable_summaries([
        ('w_conv1', W_conv1),
        ('b_conv1', b_conv1),
        ('h_conv1', h_conv1),
    ])

  tf.summary.histogram('hvonv_1', h_conv1)

  # Pooling layer - downsamples by 2X.
  # Name the 'scope' pool1 for the first pooling layer
  with tf.name_scope('pool1'):
    # use the max_pool_2x2 function to create a pooling layer after the first
    # convolution layer
    h_pool1 = max_pool_2x2(h_conv1)
  tf.summary.histogram('hpool_1', h_pool1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  # name the 'scope' conv2 for the second convolutional layer
  with tf.name_scope('conv2'):
    # use the weight_variable helper function to create a 5 X 5 convolutional
    # layer from the 32 pooled channels to 64 new feature channels
    W_conv2 = weight_variable([5, 5, 32, 64])

    # The bias is just the size of the output channel
    b_conv2 = bias_variable([64])

    # compute the relu activation for the second convolutional layer
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    create_variable_summaries([
        ('w_conv2', W_conv2),
        ('b_conv2', b_conv2),
        ('h_conv2', h_conv2),
    ])
    tf.summary.histogram('h_conv2', h_conv2)

  # Second pooling layer.
  # name the scope to display this later in the graph
  with tf.name_scope('pool2'):
    # create the second 2 X 2 pooling layer
    h_pool2 = max_pool_2x2(h_conv2)
    tf.summary.histogram('h_pool2', h_pool2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  # We name this 'fc1' for 'fully connected layer 1' in the graph
  with tf.name_scope('fc1'):
    # as the cookie cutter documentation says, this weight variable will be 7 X
    # 7 X 64 feature maps.  We get 7 by dividing 28 by 2 twice, once for each
    # pooling layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])

    # the bias is just the size of the output layer
    b_fc1 = bias_variable([1024])
    create_variable_summaries([
        ('w_fc1', W_fc1),
        ('b_fc1', b_fc1),
    ])

    # We can flatten the pooling layer so that the input to the fully connected
    # layer is just a 1 X # of remaining pixels matrix.  We use -1 to allow
    # tensorflow to figure out the number layers
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    # We now compute the relu for the fully connected layer
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    tf.summary.histogram('h_fc1', h_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  # Declare that dropout occurs here on the tf graph
  with tf.name_scope('dropout'):
    # create an input variable for the dropout probability
    keep_prob = tf.placeholder(tf.float32)

    # use dropout at the fully connected layer to prevent overfitting
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


  # Map the 1024 features to 10 classes, one for each digit
  # Name this scope fc2 for the second fully connected layer, or the output
  # layer
  with tf.name_scope('fc2'):
    # The weight is 1024 X 10, since we have 1024 inputs from the fc1 and we
    # want 10 outputs, representing the network's prediction that the input
    # pixels represent the given output digit
    W_fc2 = weight_variable([1024, 10])

    # The bias is just the size of the output vector
    b_fc2 = bias_variable([10])

    # the prediction of the convolution layer is just the output of the
    # previous layer times the weight of this layer plus the bias
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    create_variable_summaries([
        ('w_fc2', W_fc2),
        ('b_fc2', b_fc2),
    ])

  return y_conv, keep_prob


def main(_, *args, **kwargs):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  # create an input variable for x, our image.  The number of images in the
  # input can be of any size
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  # create an input variable for the labeled dataset.  There can be any number
  # of labels based on the input size, but they must have an output vector of
  # 10 for each digit
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  # set this part of the graph to 'loss' because we are computing the loss
  # function here
  with tf.name_scope('loss'):
    with tf.name_scope('cross_entropy'):
      # Use tf's magic cross entropy function to compute the ce loss between the
      # predicted digits and the labeled digits
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                              logits=y_conv)
    with tf.name_scope('cross_entropy_mean'):
      # get the average cross entropy loss in this layer accross the input examples
      cross_entropy = tf.reduce_mean(cross_entropy)
      tf.summary.scalar('cross_entropy', cross_entropy)

  # use the adam optimizer to train our neural network, since it is better and
  # faster than regular backpropagation
  with tf.name_scope('adam_optimizer'):
    # our goal is to minimize the cross entropy
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  # Name this part of the graph accuracy - we will be calculating our accuracy
  # on the training set here
  with tf.name_scope('accuracy'):
    # we will know if our prediction is correct if the highest output vector on
    # the final layer is the same as what the label set suggests
    with tf.name_scope('correct'):
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

      # We will cast our prediction to a float from a boolean so we can compute
      # the accuracy
      correct_prediction = tf.cast(correct_prediction, tf.float32)
    with tf.name_scope('accuracy'):

      # The accuracy is just the total predicted correct / the size of the dataset.
      # Since the correct prediction vector is 0's and 1's this will perform that
      # calculation
      accuracy = tf.reduce_mean(correct_prediction)
      tf.summary.scalar('accuracy', accuracy)

  # create a location to create and store the graph info
  merged = tf.summary.merge_all()

  # start the session - now the actual computation happens
  with tf.Session() as sess:
    # Merge all summary info here, so we can write it out.
    train_writer = tf.summary.FileWriter(FLAGS.data_dir + '/tmp/tf/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.data_dir + '/tmp/tf/test')
    # initialize all of the random variables
    sess.run(tf.global_variables_initializer())

    # loop over the epocs
    for i in range(1000):
      # get the next 50 examples for the batch
      batch = mnist.train.next_batch(50)

      # gather performance metrics every 100 epocs
      if i % 100 == 0:
        # evaluate the test set accuracy.  We use a keep_prob of 1 because
        # we don't want to drop any of the neurons during evaluation
        summary, test_accuracy = sess.run(
            [merged, accuracy],
            feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
        )
        # output the training accuracy
        print('step %d, test accuracy %g' % (i, test_accuracy))
        test_writer.add_summary(summary, i)
      # run feedforward and backpropagation on the batch.  We use a keep
      # probability of .5, meaning we remove a neurons influence on the output
      # randomly half the time
      summary, _ = sess.run(
          [merged, train_step],
          feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}
      )
      train_writer.add_summary(summary, i)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
