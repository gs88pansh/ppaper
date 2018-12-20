import tensorflow as tf
i = tf.Variable(2, dtype=tf.float32)
ra = tf.placeholder(dtype=tf.int32)

cost = tf.pow(i, 4)


def cond(a, n):
    return  n < ra
def body(a, n):
    n = n + 1
    cost = tf.multiply(a, a)
    return cost, n

b, v = tf.while_loop(cond, body, [i, tf.constant(0)])

i_grad=tf.gradients(b,[i])




with tf.Session() as sess:
    tf.global_variables_initializer().run()
    feed = {ra : 1}
    print(sess.run(i_grad, feed_dict=feed))
