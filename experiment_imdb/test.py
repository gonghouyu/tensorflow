import tensorflow as tf

W_concat_1 = tf.get_variable("att_w_concat_1",[1])

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(W_concat_1))
