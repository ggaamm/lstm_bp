from __future__ import print_function

import pred_input_data
traces = None
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import os

trace_directory = '/Users/gorkeralp/Dropbox/Okul/Research/branchprediction_champ2016/tensorflow_test/'
model_directory = '/Users/gorkeralp/Downloads/'

# Parameters
learning_rate = 0.001
training_iters = 1000
batch_size = 500
display_step = 100
number_of_layers = 8

# Network Parameters
n_input = 2 # MNIST data input [0,1] or [1,0]
n_steps = 25 # timesteps
n_hidden = 25 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input]) #sequences * input
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate = tf.placeholder("float", [None, 2*n_hidden*number_of_layers])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

weights_temp = {
    'hidden': np.zeros((n_input, n_hidden)) , # Hidden layer weights
    'out': np.zeros((n_hidden, n_classes))
}
biases_temp = {
    'hidden': np.zeros((n_hidden)),  # Hidden layer weights
    'out': np.zeros((n_classes))
}

matrixArr = np.zeros((2*n_hidden,4*n_hidden))

def RNN(_X, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm1 = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = rnn_cell.MultiRNNCell([lstm1] * number_of_layers)
    #print '{0:2d} {1:3d} {2:4d}'.format(lstm_cell.input_size, lstm_cell.output_size, lstm_cell.state_size)

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = RNN(x, istate, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1)) # ?,2 -> argmax 1 means choose bigger 1,0 or 0,1 for each batch
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver()
count = 0
# Launch the graph
with tf.Session() as sess: #config=tf.ConfigProto(log_device_placement=True)
    #with tf.device("/gpu:0"):
    sess.run(init)
    #print weights['hidden'].eval()

    step = 1
    # Keep training until reach max iterations
    for filename in os.listdir(trace_directory):
        if filename[0] is '.':
            continue
        try:
            print ("restoring model file")
            saver.restore(sess, model_directory + 'model.ckpt')
        except:
            print ("restore file doesn't exist, continue without it")
        print ("file: {0} is being trained".format(filename))
        traces = pred_input_data.read_data_sets(trace_directory+filename,trace_directory+filename)
        print ("file: {0} is read".format(filename))

        while step * batch_size < traces.train.input_traces.shape[0]:
            batch_xss, batch_ys = traces.train.next_batch(batch_size,timesteps=n_steps)
            ## print(batch_xss.shape)
            #mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_xs = batch_xss.reshape((batch_size, n_steps, n_input))
            # Fit training using batch data

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                           istate: np.zeros((batch_size, 2*n_hidden*number_of_layers))})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                    istate: np.zeros((batch_size, 2*n_hidden*number_of_layers))})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                                 istate: np.zeros((batch_size, 2*n_hidden*number_of_layers))})
                print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                      ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1
        print ("Optimization Finished!")
        count += 1
        print (count)
        #weightss = tf.get_variable('Matrix')
        #weightss.eval()
        #print weights['hidden'].get_shape()
        #weights_temp['hidden'] = (weights_temp['hidden'] + weights['hidden'].eval())/count
        #weights_temp['out'] = (weights_temp['out'] + weights['out'].eval()) / count
        #biases_temp['hidden'] = ( biases_temp['hidden'] + biases['hidden'].eval()) / count
        #biases_temp['out'] = (biases_temp['out'] + biases['out'].eval()) / count
        save_path = saver.save(sess, model_directory + "model.ckpt")
        print("Model saved in file: %s" % save_path)
        '''
        with tf.variable_scope("RNN", reuse=True):
            with tf.variable_scope("BasicLSTMCell", reuse=True):
                with tf.variable_scope("Linear", reuse=True):
                    v1 = tf.get_variable("Matrix", [2 * n_hidden, 4 * n_hidden])
                    matrixArr = (v1.eval()) / count
                    #print('++')
                    #print('++')
'''
    for filename in os.listdir(trace_directory):
        print (filename)
        traces1 = pred_input_data.read_data_sets(trace_directory+filename, trace_directory+filename)
        test_len = 50
        test_data = traces1.test.input_traces[:n_hidden*test_len]
        test_data = test_data.reshape(-1, n_steps, n_input)
        test_label = traces1.test.result_traces[:n_hidden*test_len]
        test_label = test_label[::n_steps, :]
        # test_label = traces.test.labels[np.ix_(rows), :]
        print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                           istate: np.zeros((test_len, 2 * n_hidden*number_of_layers))}))

"""
    #print count
    print ('----')
    print (weights_temp['hidden'])
    print ('----')
    print (weights_temp['out'])
    print ('----')
    print (biases_temp['hidden'])
    print ('----')
    print (biases_temp['out'])
    print ('--+-')
    print (matrixArr)
    print ('--+-')
    print (weights_temp['hidden'].shape)
    print ('----')
    print (weights_temp['out'].shape)
    print ('----')
    print (biases_temp['hidden'].shape)
    print ('----')
    print (biases_temp['out'].shape)
    print ('----')
    print (matrixArr.shape)
    # Calculate accuracy for 256 mnist test images
    # Calculate accuracy for 256 mnist test images

"""


