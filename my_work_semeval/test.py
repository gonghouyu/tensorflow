import tensorflow as tf

x = tf.constant([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
idx = tf.constant([1, 0, 2])
idx_flattened = tf.range(0, x.shape[0]) * x.shape[1] + idx
y = tf.gather(tf.reshape(x, [-1]),  # flatten input
              idx_flattened)  # use flattened indices

with tf.Session(''):
  print y.eval()  # [2 4 9]
