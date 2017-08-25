import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn, rnn_cell
from semval2016_data import *
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops


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


# Network Parameters
learning_rate = 0.0001
learning_rate_decay = 0.80
learning_rate_min = 0.0000001
training_iters = 2000000
batch_size = 128
display_step = 200

n_input = 200
n_steps = 40
n_hidden = 120
n_classes = 3


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
#state = tf.placeholder("float", shape=[None, n_hidden])


# define paramaters
with tf.variable_scope("input"):
    input_w = tf.get_variable("i_w",[n_input, n_hidden])
    input_b = tf.get_variable("i_b",[n_hidden]) 

with tf.variable_scope("hidden"):
    hidden_w = tf.get_variable("h_w",[n_hidden, n_hidden])
    hidden_b = tf.get_variable("h_b",[n_hidden]) 

with tf.variable_scope("bias_all"):
    bias_all = tf.get_variable("b_all",[n_hidden],initializer=init_ops.constant_initializer(0.0))

output_w = {
    'out_w': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    #'out_w': tf.get_variable("o_w",[n_hidden, n_classes])
}
output_b = {
    'out_b': tf.Variable(tf.random_normal([n_classes]))
    #'out_b': tf.get_variable("o_b",[n_classes])
}

# Construct network
#def RNN(x,state):
def RNN(x,output_w,output_b):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)			# x:(40 list,every list--120*200 tensor)

    batch_size = array_ops.shape(x[0])[0]
    state = array_ops.zeros(
        array_ops.pack([batch_size, 120]), dtype="float")
    state.set_shape([None, 120])

    #print(tf.Session().run(state))
    #exit()
    outputs = []
    with tf.variable_scope("RNN_network") as scope:
        for time_step in range(n_steps):
            if time_step > 0: scope.reuse_variables()

            WI_i = tf.matmul(x[time_step], input_w)

            WH_i = tf.matmul(state, hidden_w)

            y_update = tf.add(WH_i, WI_i) + bias_all
            y_update = tf.add(WH_i, WI_i)
            y_update = tanh(y_update)

            state = y_update
            #print(state.eval(tf.Session()))
            #exit()
            outputs.append(y_update)
        output = tf.matmul(outputs[-1], output_w['out_w'])
        output = tf.nn.bias_add(output, output_b['out_b'])
        return output,outputs

# pred = RNN(x, weights, biases)
#pred,outputs = RNN(x,state)

pred,outputs_2 = RNN(x,output_w,output_b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    step_all = 0
    # Keep training until reach max iterations
    while step_all * batch_size < training_iters:

        index_start = step*batch_size
        index_end   = index_start+batch_size

        batch_x = X_train[index_start:index_end,]
        batch_y = y_train[index_start:index_end,]
        batch_state = np.array(np.zeros([batch_size,n_hidden],dtype = float))
        batch_validation_state = np.array(np.zeros([len(X_validation),n_hidden],dtype = float))
        '''
        print(sess.run(input_w))

        for i in range(0,40):
            print(sess.run(outputs_2[i][0:1,0:7], feed_dict={x: batch_x, y: batch_y}))
        exit()
        '''
        #sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, state:batch_state})
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step_all % display_step == 0:

            #acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, state:batch_state})
            #loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, state:batch_state})
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
        
        if step_all % display_step == 0:
            #acc_v = sess.run(accuracy,feed_dict={x: X_validation ,y: y_validation, state:batch_validation_state})
            acc_v = sess.run(accuracy,feed_dict={x: X_validation ,y: y_validation})
            print("Validation accuracy = %.5f" % (acc_v))


        if step_all % display_step == 0:
            learning_rate = learning_rate * learning_rate_decay
            if learning_rate < learning_rate_min:
                learning_rate = learning_rate_min
            
    print("Optimization Finished!")


