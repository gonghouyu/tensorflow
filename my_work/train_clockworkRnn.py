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
learning_rate = 0.001
learning_rate_decay = 0.80
learning_rate_min = 0.0000001
training_iters = 2000000
batch_size = 128
display_step = 200

n_input = 200
n_steps = 40
n_hidden = 40
n_classes = 3

# Clockwork RNN parameters
periods     = [1,2,4,8,16,40]


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
state = tf.placeholder("float", shape=[None, n_hidden])

# Define weights
# clockwork_mask = tf.constant(np.triu(np.ones([n_hidden, n_hidden])), dtype=tf.float32, name="mask")

module_num = n_hidden//n_steps
mask_tril = np.zeros([n_hidden,n_hidden])
mask_triu = np.ones([n_hidden,n_hidden])

for i in range(n_hidden):
    for j in range(n_hidden):
        k = ((i//module_num)+1)*module_num-1
        t = ((i//module_num)+1)*module_num-4
        if j <= k:
            mask_tril[i][j] = 1
        if j <= t:
            mask_triu[i][j] = 0


#clockwork_mask = tf.constant(mask_tril, dtype=tf.float32, name="mask")
clockwork_mask = tf.constant(mask_triu, dtype=tf.float32, name="mask")


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


# group_index = 0


# Construct network
def RNN(x,state):

    activation_hidden = tf.tanh

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)			# x:(40 list,every list--120*200 tensor)
    
    output_state = []
    group_index = 0
    with tf.variable_scope("clockwork_cell") as scope:

        for time_step in range(n_steps):
            if time_step > 0: scope.reuse_variables()
            
            for i in range(len(periods)):
                if time_step % periods[i] == 0:
                    group_index = periods[i]
            
            #group_index = n_steps
            input_w = tf.concat(1,input_list_w[0:group_index])
            input_b = tf.concat(0,input_list_b[0:group_index])
            WI_x = tf.matmul(x[time_step], input_w)
            #WI_x = tf.nn.bias_add(WI_x, input_b, name="WI_x")

            #hidden_w = tf.concat(1,hidden_list_w)
            hidden_w_all = tf.concat(1,hidden_list_w)
            hidden_mask = tf.mul(hidden_w_all,clockwork_mask)
            hidden_w = hidden_mask[:,0:(group_index*n_hidden//n_steps)]
            hidden_b = tf.concat(0,hidden_list_b[0:group_index])
            WH_y = tf.matmul(state, hidden_w)
            #WH_y = tf.nn.bias_add(WH_y, hidden_b, name="WH_y")

            y_update = tf.add(WH_y, WI_x, name="state")
            y_update = tanh(y_update)


            state = tf.concat(1, [y_update, state[:,(group_index*n_hidden//n_steps):n_hidden]])
            output_state.append(state)

        output_list = []
        # for i in range(group_index):
        for i in range(n_steps):
            output_list.append(output_list_w[i])


        output_w = tf.concat(0,output_list)
        predictions = tf.matmul(output_state[-1], output_w)
        predictions = tf.nn.bias_add(predictions, output_b['out_b'])
        return predictions

# pred = RNN(x, weights, biases)
pred = RNN(x,state)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

'''
D_iw, D_ib, D_hw, D_hb, D_ow, D_ob = tf.gradients(cost,
                                     [input_list_w,
                                      input_list_b,
                                      hidden_list_w,
                                      hidden_list_b,
                                      output_list_w,
                                      output_list_b])
'''
'''
D_iw = tf.gradients(cost,input_list_w)
D_ib = tf.gradients(cost,input_list_b)
D_hw = tf.gradients(cost,hidden_list_w)
D_hb = tf.gradients(cost,hidden_list_b)
D_ow = tf.gradients(cost,output_list_w)
D_ob = tf.gradients(cost,output_list_b)
'''

all_list = []
all_list.extend(input_list_w)
all_list.extend(input_list_b)
all_list.extend(hidden_list_w)
all_list.extend(hidden_list_b)
all_list.extend(output_list_w)
all_list.extend([output_b['out_b']])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=all_list)


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

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, state:batch_state})

        if step_all % display_step == 0:

            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, state:batch_state})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, state:batch_state})
            print("Iter " + str(step_all*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)  + ", Learning rate = " + \
                  "{:.10f}".format(learning_rate))

        step += 1
        step_all += 1

        if step == int(math.floor(len(X_train)/batch_size)):
            step = 0
        
        if step_all % display_step == 0:
            acc_v = sess.run(accuracy,feed_dict={x: X_validation ,y: y_validation, state:batch_validation_state})
            print("Validation accuracy = %.5f" % (acc_v))


        if step_all % display_step == 0:
            learning_rate = learning_rate * learning_rate_decay
            if learning_rate < learning_rate_min:
                learning_rate = learning_rate_min
            
    print("Optimization Finished!")


