import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn, rnn_cell
from semval2016_data import *
from tensorflow.python.ops import array_ops
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
learning_rate = 0.0003
learning_rate_decay = 0.95
learning_rate_min = 0.000001
training_iters = 100000
batch_size = 32
display_step = 1000
decay_step = 200


n_input = 200
n_steps = 30
n_hidden = 40
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
    W_h = tf.get_variable("att_w_1",[2*n_hidden, 2*n_hidden])
    w = tf.get_variable("att_w_2",[2*n_hidden, 1])
    W_p = tf.get_variable("att_w_3",[1])
    W_x = tf.get_variable("att_w_4",[1])


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
    a = tf.nn.softmax(tf.matmul(M,w))    # (N*batch) * 1
    a = tf.reshape(a, [n_steps,-1, 1])    # N*batch*1
    a = tf.transpose(a, [1,2,0])    # batch*1*N


    outputs_all = tf.reshape(outputs_all, [n_steps,-1, 2*n_hidden])    # N*batch*d
    outputs_all = tf.transpose(outputs_all, [1,0,2])    # batch*N*d

    batch_s = 32

    a = tf.split(0, batch_s, a)
    outputs_all = tf.split(0, batch_s, outputs_all)

    r = []
    for i in range(batch_s):
        a_2 = a[i][0:1,:,:]
        o_2 = outputs_all[i][0:1,:,:]
        att = tf.reshape(a_2,[1, n_steps])
        out = tf.reshape(o_2,[n_steps,2*n_hidden])
        r.append(tf.matmul(att,out))
    r = tf.concat(0,r)    # batch*d



    #attention_4 = tf.matmul(r,a)
    _h = tanh(W_p*r + W_x*outputs[-1])

    predict = tf.matmul(_h, weights['out']) + biases['out']
    return predict,outputs


#temp = RNN(x, weights, biases)


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
    writer = tf.train.SummaryWriter('tensorboard/bilstm_attention', sess.graph)
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
        '''
        if step_all % show_step == 0:
            result = sess.run(merged,feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(result, step_all)
        '''

        if step_all % display_step == 0:
            '''
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step_all*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)  + ", Learning rate = " + \
                  "{:.10f}".format(learning_rate))
            '''
            # validation accuracy
            acc_validation_all = 0
            step_v = 0
            for validation_step in range(int(num_validation//batch_size)):

                index_start_v = validation_step*batch_size
                index_end_v   = index_start_v+batch_size

                acc_v = sess.run(accuracy,feed_dict={x:  X_validation[index_start_v:index_end_v,],y: y_validation[index_start_v:index_end_v,]})

                acc_validation_all += acc_v
                step_v += 1
            accuracy_validation = acc_validation_all/step_v
            '''
            acc_v = sess.run(accuracy,feed_dict={x: X_validation ,y: y_validation,})
            print("Validation accuracy = %.5f" % (acc_v))
            '''

            # train accuracy,loss
            acc_train_all = 0
            loss_train_all = 0
            step_a = 0
            for train_step in range(int(num_train//batch_size)):

                index_start_a = train_step * batch_size
                index_end_a   = index_start_a + batch_size

                acc_a = sess.run(accuracy,feed_dict={x:  X_train[index_start_a:index_end_a,],y: y_train[index_start_a:index_end_a,]})
                loss_a = sess.run(cost,feed_dict={x:  X_train[index_start_a:index_end_a,],y: y_train[index_start_a:index_end_a,]})
                loss_train_all += loss_a
                acc_train_all += acc_a
                step_a += 1
            accuracy_train = acc_train_all/step_a

            result = sess.run(merged,feed_dict={
                                     x: batch_x, 
                                     y: batch_y,
                                     validation_accuracy: accuracy_validation,
                                     train_accuracy: accuracy_train,
                                     train_loss: loss_train_all})
            writer.add_summary(result, step_all)

            print("Iter " + str(step_all*batch_size) + ", Train Loss= " + \
                  "{:.6f}".format(loss_train_all) + ", Training Accuracy= " + \
                  "{:.5f}".format(accuracy_train)  + ", Validation Accuracy= " + \
                  "{:.5f}".format(accuracy_validation) + ", Learning rate = " + \
                  "{:.8f}".format(learning_rate))


        # learning rate
        if step_all % decay_step == 0:
            learning_rate = learning_rate * learning_rate_decay
            if learning_rate < learning_rate_min:
                learning_rate = learning_rate_min

        # change steps
        step += 1
        step_all += 1
        if step == int(math.floor(len(X_train)/batch_size)):
            step = 0
    print("Optimization Finished!")


