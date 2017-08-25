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
learning_rate = 0.0005
learning_rate_decay = 0.90
learning_rate_min = 0.00001
training_iters = 1000000
batch_size = 50
display_step = 500
decay_step = 250

n_input = 200
n_steps = 20
n_hidden = 40
n_classes = 3


# Clockwork RNN parameters
periods     = [1,2,4,8,16,20]


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
#state = tf.placeholder("float", shape=[None, n_hidden])



# Define weights
module_num = n_hidden//n_steps
mask_tril = np.zeros([n_hidden,n_hidden])
mask_triu = np.ones([n_hidden,n_hidden])

for i in range(n_hidden):
    for j in range(n_hidden):
        k = ((i//module_num)+1)*module_num-1
        t = ((i//module_num)+1)*module_num-(module_num+1)
        if j <= k:
            mask_tril[i][j] = 1
        if j <= t:
            mask_triu[i][j] = 0


clockwork_mask = tf.constant(mask_tril, dtype=tf.float32, name="mask")
#clockwork_mask = tf.constant(mask_triu, dtype=tf.float32, name="mask")




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
        array_ops.pack([batch_size, n_hidden]), dtype="float")
    state.set_shape([None, n_hidden])

    outputs = []
    with tf.variable_scope("RNN_network") as scope:

        for time_step in range(n_steps):
            if time_step > 0: scope.reuse_variables()

            for i in range(len(periods)):
                if time_step % periods[i] == 0:
                    group_index = periods[i]

            WI_i = tf.matmul(x[time_step], input_w[:,0:(group_index*n_hidden//n_steps)])
            hidden_mask = tf.mul(hidden_w,clockwork_mask)
            WH_i = tf.matmul(state, hidden_mask[:,0:(group_index*n_hidden//n_steps)])

            y_update = tf.add(WH_i, WI_i)
            y_update = tanh(y_update)

            state = tf.concat(1, [y_update, state[:,(group_index*n_hidden//n_steps):n_hidden]])
            outputs.append(state)
        output = tf.matmul(outputs[-1], output_w['out_w'])
        output = tf.nn.bias_add(output, output_b['out_b'])
        return output,outputs



pred,outputs_2 = RNN(x,output_w,output_b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


validation_accuracy = tf.placeholder("float")
tf.scalar_summary('validation_accuracy', validation_accuracy)
train_accuracy = tf.placeholder("float")
tf.scalar_summary('train_accuracy', train_accuracy)
train_loss = tf.placeholder("float")
tf.scalar_summary('train_loss', train_loss)


init = tf.initialize_all_variables()

with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('tensorboard/clockrnn/exp_1', sess.graph)
    sess.run(init)
    step = 0
    step_all = 0
    # train
    while step_all * batch_size <= training_iters:
        index_start = step*batch_size
        index_end   = index_start+batch_size
        batch_x = X_train[index_start:index_end]
        batch_y = y_train[index_start:index_end]

        '''
        print(sess.run(outputs_2[50][0:1,0:200], feed_dict={x: batch_x, y: batch_y}))
        print(sess.run(_h[0:1,0:200], feed_dict={x: batch_x, y: batch_y}))
        exit()
        '''

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        # display_step
        if step_all % display_step == 0:
            # validation
            acc_validation_all = 0
            step_v = 0
            for validation_step in range(int(num_validation//batch_size)):
                index_start_v = validation_step*batch_size
                index_end_v   = index_start_v+batch_size
                acc_v = sess.run(accuracy,feed_dict={x:X_validation[index_start_v:index_end_v],y: y_validation[index_start_v:index_end_v]})
                acc_validation_all += acc_v
                step_v += 1
            accuracy_validation = acc_validation_all/step_v
            # train
            acc_train_batch = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss_train_batch = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            # summary
            result = sess.run(merged,feed_dict={
                                     x: batch_x, 
                                     y: batch_y,
                                     validation_accuracy: accuracy_validation,
                                     train_accuracy: acc_train_batch,
                                     train_loss: loss_train_batch})
            writer.add_summary(result, step_all)
            print("Validation accuracy = %.5f" % (accuracy_validation))
            print("Iter " + str(step_all*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss_train_batch) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc_train_batch)  + ", Learning rate = " + \
                  "{:.10f}".format(learning_rate))

        # learning rate
        if (step_all % decay_step == 0 and step_all != 0):
            learning_rate = learning_rate * learning_rate_decay
            if learning_rate < learning_rate_min:
                learning_rate = learning_rate_min

        # change steps
        step += 1
        step_all += 1
        if step == int(math.floor(len(X_train)/batch_size)):
            step = 0
    print("Optimization Finished!")


