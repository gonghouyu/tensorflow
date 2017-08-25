import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn, rnn_cell
from semval2016_data import *
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
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
learning_rate = 0.0003
learning_rate_decay = 0.95
learning_rate_min = 0.000001
training_iters = 2000000
batch_size = 128
display_step = 200

n_input = 200
n_steps = 30
n_hidden = 40
n_classes = 3
attention_num = 40


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
state_c = tf.placeholder("float", shape=[None, n_hidden])
state_h = tf.placeholder("float", shape=[None, n_hidden])


# define paramaters
with tf.variable_scope("input_i"):
    input_w_i = tf.get_variable("i_i_w",[n_input, n_hidden])
with tf.variable_scope("input_j"):
    input_w_j = tf.get_variable("i_j_w",[n_input, n_hidden])
with tf.variable_scope("input_f"):
    input_w_f = tf.get_variable("i_f_w",[n_input, n_hidden])
with tf.variable_scope("input_o"):
    input_w_o = tf.get_variable("i_o_w",[n_input, n_hidden])
with tf.variable_scope("hidden_i"):
    hidden_w_i = tf.get_variable("h_i_w",[n_hidden, n_hidden])
with tf.variable_scope("hidden_j"):
    hidden_w_j = tf.get_variable("h_j_w",[n_hidden, n_hidden])
with tf.variable_scope("hidden_f"):
    hidden_w_f = tf.get_variable("h_f_w",[n_hidden, n_hidden])
with tf.variable_scope("hidden_o"):
    hidden_w_o = tf.get_variable("h_o_w",[n_hidden, n_hidden])

with tf.variable_scope("bias_all"):
    bias_all = tf.get_variable("b_all",[n_hidden],initializer=init_ops.constant_initializer(0.0))

'''
with tf.variable_scope("attention_all"):
    attention_w = tf.get_variable("attention_w",[n_steps])
'''

attention_w = []
for i in range(n_steps):
    attention_w.append(tf.get_variable("attention_w_"+str(i),[1]))

output_w = {
    'out_w': tf.Variable(tf.random_normal([n_hidden+attention_num, n_classes]))
}
output_b = {
    'out_b': tf.Variable(tf.random_normal([n_classes]))
}


'''
with tf.variable_scope("output"):
    output_w = tf.get_variable("o_w",[n_hidden, n_classes])
    output_b = tf.get_variable("o_b",[n_classes])
'''

# Construct network
def LSTM(x,state_c,state_h):

    # activation_hidden = tf.tanh
    # activation_output = tf.nn.relu

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)			# x:(40 list,every list--120*200 tensor)

    outputs_c = []
    outputs_h = []
    with tf.variable_scope("LSTM_network") as scope:
        for time_step in range(n_steps):
            if time_step > 0: scope.reuse_variables()

            WI_i = tf.matmul(x[time_step], input_w_i)
            WI_j = tf.matmul(x[time_step], input_w_j)
            WI_f = tf.matmul(x[time_step], input_w_f)
            WI_o = tf.matmul(x[time_step], input_w_o)

            WH_i = tf.matmul(state_h, hidden_w_i)
            WH_j = tf.matmul(state_h, hidden_w_j)
            WH_f = tf.matmul(state_h, hidden_w_f)
            WH_o = tf.matmul(state_h, hidden_w_o)

            i = tf.add(WH_i, WI_i) + bias_all
            j = tf.add(WH_j, WI_j) + bias_all
            f = tf.add(WH_f, WI_f) + bias_all
            o = tf.add(WH_o, WI_o) + bias_all
            # y_update = activation_hidden(y_update)
            state_c = state_c * sigmoid(f + 1.0) + sigmoid(i) * tanh(j)
            state_h = tanh(state_c) * sigmoid(o)
            outputs_c.append(state_c)
            outputs_h.append(state_h)

            
            '''
            output_step = tf.matmul(outputs_h[-1], output_w['out_w'])
            output_step = tf.nn.bias_add(output_step, output_b['out_b'])

            cost_step = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_step, y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_step)
            '''
        #print(attention_w)
        #print(outputs_h)
        #print(tf.concat(0,outputs_h))
        #exit()
        attention_input = np.dot(attention_w,x) 
        #attention_input = attention_input * 0.01
        
        attention_input_40 = tf.concat(1,[attention_input[:,0:1],attention_input[:,3:4]])
        for i in range(2,attention_num):
            attention_input_40 = tf.concat(1,[attention_input_40,attention_input[:,(i*3):(i*3+1)]])
        attention_input_40 = attention_input_40 * 0.1
        #output_final = attention_input_40 + outputs_h[-1]
        output_final = tf.concat(1,[attention_input_40,outputs_h[-1]])
        output = tf.matmul(output_final, output_w['out_w'])
        output = tf.nn.bias_add(output, output_b['out_b'])
        return output,outputs_h,attention_input_40

# pred = RNN(x, weights, biases)
pred,outputs_all,attention_input = LSTM(x,state_c,state_h)
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
        print(sess.run(attention_input[0:1,:], feed_dict={x: batch_x, y: batch_y, state_c:batch_state, state_h:batch_state}))
        #print(sess.run(input_w_i))
        for i in range(0,n_steps):
            print(sess.run(outputs_all[i][0:1,0:6], feed_dict={x: batch_x, y: batch_y, state_c:batch_state, state_h:batch_state}))
        exit()
        '''
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, state_c:batch_state, state_h:batch_state})

        if step_all % display_step == 0:

            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, state_c:batch_state, state_h:batch_state})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, state_c:batch_state, state_h:batch_state})
            print("Iter " + str(step_all*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)  + ", Learning rate = " + \
                  "{:.10f}".format(learning_rate))

        step += 1
        step_all += 1

        if step == int(math.floor(len(X_train)/batch_size)):
            step = 0
        
        if step_all % display_step == 0:
            acc_v = sess.run(accuracy,feed_dict={x: X_validation ,y: y_validation, state_c:batch_validation_state, state_h:batch_validation_state})
            print("Validation accuracy = %.5f" % (acc_v))


        if step_all % display_step == 0:
            learning_rate = learning_rate * learning_rate_decay
            if learning_rate < learning_rate_min:
                learning_rate = learning_rate_min
            
    print("Optimization Finished!")


