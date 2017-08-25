import tensorflow as tf
'''
a = "aaa"
print("sss"+a)
'''

x = [0,1,2,3,4,5,6,7,8,9,10]
print(x[0:5])

for i in range(1,41,1):
    print(i)
    with tf.variable_scope("input"+str(i)):
        x[i] = tf.Variable(tf.random_normal([2, 2]))


#print(x[0].ref())
#print(x[1].ref())
#print(x[2].ref())

