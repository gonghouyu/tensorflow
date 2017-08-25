import imdb_data
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn, rnn_cell



# Load the training data
X_train,X_validation,y_train,y_validation = imdb_data.load_imdb_data()
num_train      = len(X_train)
num_validation = len(X_validation)
print(num_train)
print(num_validation)



# Network Parameters
learning_rate = 0.001
learning_rate_decay = 0.80
learning_rate_min = 0.000001
training_iters = 600000
batch_size = 100
display_step = 1000
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
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct network
def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)			# x:(40 list,every list--120*200 tensor)
    gru_cell = rnn_cell.GRUCell(n_hidden)
    outputs, states = rnn.rnn(gru_cell, x, dtype=tf.float32)		# outputs : 40 * batch_size * 98	# state  : batch_size * 196
    predict = tf.matmul(outputs[-1], weights['out']) + biases['out']
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
    writer = tf.train.SummaryWriter('tensorboard/imdb_gru', sess.graph)
    sess.run(init)
    step = 0
    step_all = 0
    # train
    while step_all * batch_size <= training_iters:
        index_start = step*batch_size
        index_end   = index_start+batch_size
        batch_x = X_train[index_start:index_end]
        batch_y = y_train[index_start:index_end]
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


