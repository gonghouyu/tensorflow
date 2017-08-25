from semval2016_data import *
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn, rnn_cell
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
learning_rate = 0.0005
learning_rate_decay = 0.90
learning_rate_min = 0.000001
training_iters = 500000
batch_size = 32
display_step = 500
decay_step = 100

n_input = 200
n_steps = 20
n_hidden = 50
n_classes = 3



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

with tf.variable_scope("attention_w"):
    W_h = tf.get_variable("att_w_1",[n_hidden, n_hidden])
    w = tf.get_variable("att_w_2",[n_hidden, 1])
    W_p = tf.get_variable("att_w_3",[1])
    W_x = tf.get_variable("att_w_4",[1])

with tf.variable_scope("attention_w_2"):
    W_h_2 = tf.get_variable("att_w_1_2",[n_hidden, n_hidden])
    w_2 = tf.get_variable("att_w_2_2",[n_hidden, 1])
    W_p_2 = tf.get_variable("att_w_3_2",[1])
    W_x_2 = tf.get_variable("att_w_4_2",[1])

# Construct network
def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    gru_fw_cell = rnn_cell.GRUCell(n_hidden)
    gru_fw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_fw_cell, output_keep_prob=0.7)
    gru_bw_cell = rnn_cell.GRUCell(n_hidden)
    gru_bw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_bw_cell, output_keep_prob=0.7)
    outputs, outputs_fw, outputs_bw = rnn.bidirectional_rnn(gru_fw_cell, gru_bw_cell, x,dtype=tf.float32)    # Nlist * batch * 2hidden

    batch_s = 100
    outputs_all = tf.concat(0,outputs)          # (N*batch) * (2*n_hidden)
    outputs_fw_all = outputs_all[:,0:n_hidden]    # (N*batch) * n_hidden
    outputs_bw_all = outputs_all[:,n_hidden:2*n_hidden]    # (N*batch) * n_hidden

    # dropout
    outputs_fw_all = tf.nn.dropout(outputs_fw_all, keep_prob=0.5)
    outputs_bw_all = tf.nn.dropout(outputs_bw_all, keep_prob=0.5)

    M_fw = tanh(tf.matmul(outputs_fw_all,W_h))    # (N*batch) * 2*hidden
    M_bw = tanh(tf.matmul(outputs_fw_all,W_h_2))    # (N*batch) * 2*hidden

    # dropout
    M_fw = tf.nn.dropout(M_fw, keep_prob=0.5)
    M_bw = tf.nn.dropout(M_bw, keep_prob=0.5)

    a_fw = tf.matmul(M_fw,w)
    #a = tanh(tf.matmul(outputs_fw_all,w))
    a_fw = tf.reshape(a_fw, [n_steps,-1])    # N*batch
    a_fw = tf.transpose(a_fw, [1,0])    # batch*N
    a_fw = tf.nn.softmax(a_fw)
    a_fw = tf.reshape(a_fw, [batch_s,1,n_steps])    # batch*1*N


    a_bw = tf.matmul(M_bw,w_2)
    #a = tanh(tf.matmul(outputs_bw_all,w_2))
    a_bw = tf.reshape(a_bw, [n_steps,-1])    # N*batch
    a_bw = tf.transpose(a_bw, [1,0])    # batch*N
    a_bw = tf.nn.softmax(a_bw)
    a_bw = tf.reshape(a_bw, [batch_s,1,n_steps])    # batch*1*N



    outputs_fw_all = tf.reshape(outputs_fw_all, [n_steps,-1, n_hidden])    # N*batch*d
    outputs_fw_all = tf.transpose(outputs_fw_all, [1,0,2])    # batch*N*d
    outputs_bw_all = tf.reshape(outputs_bw_all, [n_steps,-1, n_hidden])    # N*batch*d
    outputs_bw_all = tf.transpose(outputs_bw_all, [1,0,2])    # batch*N*d

    a_fw = tf.split(0, batch_s, a_fw)
    outputs_fw_all = tf.split(0, batch_s, outputs_fw_all)
    a_bw = tf.split(0, batch_s, a_bw)
    outputs_bw_all = tf.split(0, batch_s, outputs_bw_all)

    r_fw = []
    r_bw = []
    for i in range(batch_s):
        a_fw_2 = a_fw[i][0:1,:,:]
        o_fw_2 = outputs_fw_all[i][0:1,:,:]
        att_fw = tf.reshape(a_fw_2,[1, n_steps])            # 1*N
        out_fw = tf.reshape(o_fw_2,[n_steps,n_hidden])    # N*2*n_hidden

        a_bw_2 = a_bw[i][0:1,:,:]
        o_bw_2 = outputs_bw_all[i][0:1,:,:]
        att_bw = tf.reshape(a_bw_2,[1, n_steps])            # 1*N
        out_bw = tf.reshape(o_bw_2,[n_steps,n_hidden])    # N*2*n_hidden

        # dropout
        #att = tf.nn.dropout(att, keep_prob=0.5)
        #out = tf.nn.dropout(out, keep_prob=0.5)

        r_fw.append(tf.matmul(att_fw,out_fw))
        r_bw.append(tf.matmul(att_bw,out_bw))
    r_fw = tf.concat(0,r_fw)    # batch*d
    r_bw = tf.concat(0,r_bw)    # batch*d
    #_h = tanh(W_p*r_fw + W_p_2*r_bw + W_x*outputs[-1])
    _h_1 = tanh(W_p*r_fw + W_x*outputs_fw)
    _h_2 = tanh(W_p_2*r_bw + W_x_2*outputs_bw)
    _h = tf.concat(1,[_h_1,_h_2])

    # dropout
    _h = tf.nn.dropout(_h, keep_prob=0.25)

    predict = tf.matmul(_h, weights['out']) + biases['out']
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
    writer = tf.train.SummaryWriter('tensorboard/att_mul_dropout/att/no_dropout', sess.graph)
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


