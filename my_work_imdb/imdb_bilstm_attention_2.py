import imdb_data
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import tanh


# Load the training data
X_train,X_validation,y_train,y_validation = imdb_data.load_imdb_data()
num_train      = len(X_train)
num_validation = len(X_validation)
print(num_train)
print(num_validation)



# Network Parameters
learning_rate = 0.001
learning_rate_decay = 0.90
learning_rate_min = 0.000001
training_iters = 600000
batch_size = 32
display_step = 200
decay_step = 200
validation_display = 1000

n_input = 200
n_steps = 200
n_hidden = 128
n_classes = 2



# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

with tf.variable_scope("attention_w_one"):
    W_h = tf.get_variable("att_w_1",[2*n_hidden, 2*n_hidden])
    w = tf.get_variable("att_w_2",[2*n_hidden, 1])
    W_p = tf.get_variable("att_w_3",[1])
    W_x = tf.get_variable("att_w_4",[1])

with tf.variable_scope("attention_w_two"):
    W_2_h = tf.get_variable("att_w_two_1",[2*n_hidden, 2*n_hidden])
    w_2 = tf.get_variable("att_w_two_2",[2*n_hidden, 1])
    W_2_p = tf.get_variable("att_w_two_3",[1])


# Construct network
def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,dtype=tf.float32)
    outputs_all = tf.concat(0,outputs)    # (N*batch) * 2*n_hidden



    M = tanh(tf.matmul(outputs_all,W_h))    # (N*batch) * 2*hidden
    M_2 = tanh(tf.matmul(outputs_all,W_2_h))

    a = tf.nn.softmax(tf.matmul(M,w))    # (N*batch) * 1
    a = tf.reshape(a, [n_steps,-1, 1])    # N*batch*1
    a = tf.transpose(a, [1,2,0])    # batch*1*N
    a_2 = tf.nn.softmax(tf.matmul(M_2,w_2))    # (N*batch) * 1
    a_2 = tf.reshape(a_2, [n_steps,-1, 1])    # N*batch*1
    a_2 = tf.transpose(a_2, [1,2,0])    # batch*1*N

    outputs_all = tf.reshape(outputs_all, [n_steps,-1, 2*n_hidden])    # N*batch*d
    outputs_all = tf.transpose(outputs_all, [1,0,2])    # batch*N*d

    batch_s = 32
    a = tf.split(0, batch_s, a)
    a_2 = tf.split(0, batch_s, a_2)
    outputs_all = tf.split(0, batch_s, outputs_all)

    r = []
    r_2 = []
    for i in range(batch_s):
        a_temp = a[i][0:1,:,:]
        a_2_temp = a_2[i][0:1,:,:]
        o_temp = outputs_all[i][0:1,:,:]
        o_2_temp = outputs_all[i][0:1,:,:]
        att = tf.reshape(a_temp,[1, n_steps])
        out = tf.reshape(o_temp,[n_steps,2*n_hidden])
        att_2 = tf.reshape(a_2_temp,[1, n_steps])
        out_2 = tf.reshape(o_2_temp,[n_steps,2*n_hidden])
        r.append(tf.matmul(att,out))
        r_2.append(tf.matmul(att_2,out_2))
    r = tf.concat(0,r)    # batch*d
    r_2 = tf.concat(0,r_2)    # batch*d

    #attention_4 = tf.matmul(r,a)
    _h = tanh(W_p*r + W_2_p*r_2 + W_x*outputs[-1])

    predict = tf.matmul(_h, weights['out']) + biases['out']
    return predict,outputs


#temp = RNN(x, weights, biases)


pred,outputs_2 = RNN(x, weights, biases)
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

        batch_x = X_train[index_start:index_end]
        batch_y = y_train[index_start:index_end]

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
        if step_all % validation_display == 0:
            acc_all = 0
            step_v = 0
            for validation_step in range(int(num_validation//batch_size)):

                index_start_v = validation_step*batch_size
                index_end_v   = index_start_v+batch_size

                acc_v = sess.run(accuracy,feed_dict={x:  X_validation[index_start_v:index_end_v],y: y_validation[index_start_v:index_end_v]})

                acc_all += acc_v
                step_v += 1
            print("Validation accuracy = %.5f" % (acc_all/step_v))

            #acc_v = sess.run(accuracy,feed_dict={x: batch_v_x ,y: batch_v_y})
            #print("Validation accuracy = %.5f" % (acc_v))
        
        # learning rate
        if step_all % decay_step == 0:
            learning_rate = learning_rate * learning_rate_decay
            if learning_rate < learning_rate_min:
                learning_rate = learning_rate_min
            
    print("Optimization Finished!")


