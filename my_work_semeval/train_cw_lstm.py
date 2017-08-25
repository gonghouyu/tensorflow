import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn, rnn_cell
from semval2016_data import *
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


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
state_c = tf.placeholder("float", shape=[None, n_hidden])
state_h = tf.placeholder("float", shape=[None, n_hidden])


# Define weights
clockwork_mask = tf.constant(np.tril(np.ones([n_hidden, n_hidden])), dtype=tf.float32, name="mask")

input_list_w = [0]*(n_steps)
input_list_b = [0]*(n_steps)
hidden_list_w = [0]*(n_steps)
hidden_list_b = [0]*(n_steps)
output_list_w = []
output_w = [0]*(n_steps)
# output_list_b = [0]


# define paramaters
for i in range(n_steps):
    with tf.variable_scope("input"+str(i)):
        input_list_w[i] = tf.get_variable("input_w"+str(i)+"_"+str(i),[n_input, n_hidden//n_steps])
        input_list_b[i] = tf.get_variable("input_b"+str(i)+"_"+str(i),[n_hidden//n_steps])
    with tf.variable_scope("hidden"+str(i)):
        hidden_list_w[i] = tf.get_variable("hidden_w"+str(i)+"_"+str(i),[n_hidden, n_hidden//n_steps])
        hidden_list_b[i] = tf.get_variable("hidden_b"+str(i)+"_"+str(i),[n_hidden//n_steps])

    output_w[i] = {
        'out_w': tf.Variable(tf.random_normal([n_hidden//n_steps, n_classes]))
    }

for i in range(n_steps):
    output_list_w.append(output_w[i]['out_w'])

output_b = {
    'out_b': tf.Variable(tf.random_normal([n_classes]))
}




# Construct network
def CW_LSTM(x,state_c,state_h):

    # activation_hidden = tf.tanh
    # activation_output = tf.nn.relu

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)			# x:(40 list,every list--120*200 tensor)

    outputs_c = []
    outputs_h = []

    group_index = 0

    with tf.variable_scope("LSTM_network") as scope:
        for time_step in range(n_steps):
            if time_step > 0: scope.reuse_variables()
            for i in range(len(periods)):
                if time_step % periods[i] == 0:
                    group_index = periods[i]



            input_w = tf.concat(1,input_list_w[0:group_index])
            input_b = tf.concat(0,input_list_b[0:group_index])



            WI_i = tf.matmul(x[time_step], input_w_i)
            WI_i = tf.nn.bias_add(WI_i, input_b_i)
            WI_j = tf.matmul(x[time_step], input_w_j)
            WI_j = tf.nn.bias_add(WI_j, input_b_j)
            WI_f = tf.matmul(x[time_step], input_w_f)
            WI_f = tf.nn.bias_add(WI_f, input_b_f)
            WI_o = tf.matmul(x[time_step], input_w_o)
            WI_o = tf.nn.bias_add(WI_o, input_b_o)

            WH_i = tf.matmul(state_h, hidden_w_i)
            WH_i = tf.nn.bias_add(WH_i, hidden_b_i)
            WH_j = tf.matmul(state_h, hidden_w_j)
            WH_j = tf.nn.bias_add(WH_j, hidden_b_j)
            WH_f = tf.matmul(state_h, hidden_w_f)
            WH_f = tf.nn.bias_add(WH_f, hidden_b_f)
            WH_o = tf.matmul(state_h, hidden_w_o)
            WH_o = tf.nn.bias_add(WH_o, hidden_b_o)

            i = tf.add(WH_i, WI_i)
            j = tf.add(WH_j, WI_j)
            f = tf.add(WH_f, WI_f)
            o = tf.add(WH_o, WI_o)
            # y_update = activation_hidden(y_update)
            state_c = state_c * sigmoid(f + 1.0) + sigmoid(i) * tanh(j)
            state_h = tanh(state_c) * sigmoid(o)
            outputs_c.append(state_c)
            outputs_h.append(state_h)

            output_step = tf.matmul(outputs_h[-1], output_w['out_w'])
            output_step = tf.nn.bias_add(output_step, output_b['out_b'])

            cost_step = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_step, y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_step)



        output = tf.matmul(outputs_h[-1], output_w['out_w'])
        output = tf.nn.bias_add(output, output_b['out_b'])
        return output

# pred = RNN(x, weights, biases)
pred = CW_LSTM(x,state_c,state_h)
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


