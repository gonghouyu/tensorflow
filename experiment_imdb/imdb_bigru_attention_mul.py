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
learning_rate_decay = 0.70
learning_rate_min = 0.0001
training_iters = 500000
batch_size = 100
display_step = 500
decay_step = 250

n_input = 200
n_steps = 230
n_hidden = 100
n_classes = 2



# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}

weights_concat = {
    'out_concat': tf.Variable(tf.random_normal([2*n_hidden+n_input, n_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

with tf.variable_scope("attention_w"):
    W_h = tf.get_variable("att_w_1",[2*n_hidden, 2*n_hidden])
    w = tf.get_variable("att_w_2",[2*n_hidden, 1])
    W_p = tf.get_variable("att_w_3",[1])
    W_x = tf.get_variable("att_w_4",[1])
with tf.variable_scope("attention_w_input"):
    W_h_input = tf.get_variable("att_w_input_1",[n_input, n_input])
    w_input = tf.get_variable("att_w_input_2",[n_input, 1])
    W_p_input = tf.get_variable("att_w_input_3",[1])
    W_x_input = tf.get_variable("att_w_input_4",[n_input, 2*n_hidden])
    #W_concat_1 = tf.get_variable("att_w_concat_1",[1])
    #W_concat_2 = tf.get_variable("att_w_concat_2",[1])


# Construct network
def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)              # n_steps(list) * batch * 200
    gru_fw_cell = rnn_cell.GRUCell(n_hidden)
    gru_fw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_fw_cell, output_keep_prob=0.7)
    gru_bw_cell = rnn_cell.GRUCell(n_hidden)
    gru_bw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_bw_cell, output_keep_prob=0.7)

    outputs, _, _ = rnn.bidirectional_rnn(gru_fw_cell, gru_bw_cell, x,dtype=tf.float32)
    batch_s = 100

    outputs_all = tf.concat(0,outputs)    # (N*batch) * 2*n_hidden

    # dropout
    outputs_all = tf.nn.dropout(outputs_all, keep_prob=0.5)

    input_all = tf.concat(0,x)    # (n_steps*batch) * 2*n_hidden

    # dropout
    input_all = tf.nn.dropout(input_all, keep_prob=0.5)


    #**********************************************************************************************
    M = tanh(tf.matmul(outputs_all,W_h))    # (N*batch) * 2*hidden

    # dropout
    M = tf.nn.dropout(M, keep_prob=0.5)

    a = tf.matmul(M,w)
    a = tf.reshape(a, [n_steps,-1])    # N*batch
    a = tf.transpose(a, [1,0])    # batch*N
    a = tf.nn.softmax(a)
    a = tf.reshape(a, [batch_s,1,n_steps])    # batch*1*N

    outputs_all = tf.reshape(outputs_all, [n_steps,-1, 2*n_hidden])    # N*batch*d
    outputs_all = tf.transpose(outputs_all, [1,0,2])    # batch*N*d

    a = tf.split(0, batch_s, a)
    outputs_all = tf.split(0, batch_s, outputs_all)

    r = []
    for i in range(batch_s):
        a_temp = a[i][0:1,:,:]
        o_temp = outputs_all[i][0:1,:,:]
        att = tf.reshape(a_temp,[1, n_steps])
        out = tf.reshape(o_temp,[n_steps,2*n_hidden])

        # dropout
        att = tf.nn.dropout(att, keep_prob=0.5)
        out = tf.nn.dropout(out, keep_prob=0.5)

        r.append(tf.matmul(att,out))
    r = tf.concat(0,r)    # batch*d
    #**********************************************************************************************
    M_input = tanh(tf.matmul(input_all,W_h_input))    # (N*batch) * 2*hidden

    # dropout
    M_input = tf.nn.dropout(M_input, keep_prob=0.5)

    a_input = tf.matmul(M_input,w_input)
    #a_input = tf.matmul(input_all,w_input)

    a_input = tf.reshape(a_input, [n_steps,-1])    # N*batch
    a_input = tf.transpose(a_input, [1,0])    # batch*N
    a_input = tf.nn.softmax(a_input)
    a_input = tf.reshape(a_input, [batch_s,1,n_steps])    # batch*1*N

    '''
    a_input = tf.nn.softmax(tf.matmul(M_input,w_input))    # (N*batch) * 1
    a_input = tf.reshape(a_input, [n_steps,-1, 1])    # N*batch*1
    a_input = tf.transpose(a_input, [1,2,0])    # batch*1*N
    '''

    input_all = tf.reshape(input_all, [n_steps,-1, n_input])    # N*batch*n_input
    input_all = tf.transpose(input_all, [1,0,2])    # batch*N*n_input

    a_input = tf.split(0, batch_s, a_input)
    input_all = tf.split(0, batch_s, input_all)

    r_input = []
    for i in range(batch_s):
        a_input_temp = a_input[i][0:1,:,:]
        o_input_temp = input_all[i][0:1,:,:]
        att_input = tf.reshape(a_input_temp,[1, n_steps])
        input_input = tf.reshape(o_input_temp,[n_steps,n_input])

        # dropout
        att_input = tf.nn.dropout(att_input, keep_prob=0.5)
        input_input = tf.nn.dropout(input_input, keep_prob=0.5)

        r_input.append(tf.matmul(att_input,input_input))
    r_input = tf.concat(0,r_input)    # batch*n_input

    '''
    r_input_hidden = tanh(tf.matmul(r_input,W_x_input))
    #r_input_hidden = tf.matmul(r_input,W_x_input)
    _h = tanh(W_p*r + W_p_input*r_input_hidden + W_x*outputs[-1])
    predict = tf.matmul(_h, weights['out']) + biases['out']
    '''

    _h_temp_1 = tanh(W_p*r + W_x*outputs[-1])
    _h_temp_2 = tanh(W_p_input*r_input)
    _h_concat = tf.concat(1,[_h_temp_1,_h_temp_2])

    # dropout
    _h_concat = tf.nn.dropout(_h_concat, keep_prob=0.25)

    predict = tf.matmul(_h_concat, weights_concat['out_concat']) + biases['out']
    return predict,outputs


pred,outputs_2 = RNN(x, weights, biases)
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
    writer = tf.train.SummaryWriter('tensorboard/att_mul_dropout/mul/exp_3', sess.graph)
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
        print(sess.run(outputs_2[-1][90:91,0:200], feed_dict={x: batch_x, y: batch_y}))
        print(sess.run(r[90:91,0:200], feed_dict={x: batch_x, y: batch_y}))
        print(sess.run(r_input[90:91,0:200], feed_dict={x: batch_x, y: batch_y}))
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


