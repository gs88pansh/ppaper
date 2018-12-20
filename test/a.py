import tensorflow as tf
import numpy

BATCHSIZE=6

label=tf.expand_dims(tf.constant([0,2,3,6,7,9]),1)
index=tf.expand_dims(tf.range(0, BATCHSIZE),1)
# use a matrix
concated = tf.concat([index, label], 1)   # [[0, 0], [0, 2], [0, 3], [0, 6], [0, 7], [0, 9]] (6,2)
onehot_labels = tf.sparse_to_dense(concated, [BATCHSIZE,10], 1.0, 0.0)

# use a vector
sparse_indices2=tf.constant([1,3,4])
onehot_labels2 = tf.sparse_to_dense(sparse_indices2, [10], 1.0, 0.0)#can use

# use a scalar
sparse_indices3=tf.constant(5)
onehot_labels3 = tf.sparse_to_dense(sparse_indices3, [10], 1.0, 0.0)

sparse_tensor_00 = tf.SparseTensor(indices=[[0,0], [0,1]], values=[1,1], dense_shape=[10000,300])
dense_tensor_00 = tf.sparse_tensor_to_dense(sparse_tensor_00)

with tf.Session() as sess:
    result1=sess.run(onehot_labels)
    result2 = sess.run(onehot_labels2)
    result3 = sess.run(onehot_labels3)
    result4 = sess.run(dense_tensor_00)
    print ("This is result1:")
    print (result1)
    print ("This is result2:")
    print (result2)
    print ("This is result3:")
    print (result3)
    print ("This is result4:")
    print (result4)