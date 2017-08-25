import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn, rnn_cell
from semval2016_data import *



# Load the training data
(X_train, y_train), (X_validation, y_validation) = load_a(test_split=0.2,cnn=False,sentiment_embeddings=False,Load_Test=False)
print(len(X_train), 'train sequences')
print(len(X_validation), 'test sequences')
print('X_train shape:', X_train.shape)
print('X_validation shape:', X_validation.shape)
print('y_train shape:', y_train.shape)
print('y_validation shape:', y_validation.shape)
num_train      = X_train.shape[0]
num_validation = X_validation.shape[0]



# Parameters
learning_rate = 0.0001
learning_rate_decay = 0.80
learning_rate_min = 0.0000001
training_iters = 2000000
batch_size = 128
display_step = 200

# Network Parameters
n_input = 200
n_steps = 40
n_hidden = 120
n_classes = 3



# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])



# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct network
def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    rnn_cell_2 = rnn_cell.BasicRNNCell(n_hidden)

    outputs, states = rnn.rnn(rnn_cell_2, x, dtype=tf.float32, clock = [1, 2, 4, 8, 16, 32])
    return tf.matmul(outputs[-1], weights['out']) + biases['out']



pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    step_all = 0
    # train
    while step_all * batch_size < training_iters:

        index_start = step*batch_size
        index_end   = index_start+batch_size

        batch_x = X_train[index_start:index_end,]
        batch_y = y_train[index_start:index_end,]

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step_all % display_step == 0:

            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step_all*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)  + ", Learning rate = " + \
                  "{:.10f}".format(learning_rate))

        step += 1
        step_all += 1

        if step == int(math.floor(len(X_train)/batch_size)):
            step = 0
        # validation
        if step_all % display_step == 0:
            acc_v = sess.run(accuracy,feed_dict={x: X_validation ,y: y_validation,})
            print("Validation accuracy = %.5f" % (acc_v))
        # learning rate
        if step_all % display_step == 0:
            learning_rate = learning_rate * learning_rate_decay
            if learning_rate < learning_rate_min:
                learning_rate = learning_rate_min
            
    print("Optimization Finished!")


