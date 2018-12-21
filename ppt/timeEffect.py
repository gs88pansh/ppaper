import tensorflow as tf
import time
def f(a, b, t):
    # 计算 (exp(a)+1)/exp(a) * exp(a+bt) / (exp(a+bt)+1)
    e_a = tf.exp(a)  # exp(a)
    e_a_bt = tf.exp(tf.subtract(a,tf.multiply(b,t)))  # exp(a-bt)
    e_a_plus1 =  (e_a_bt) / tf.add(e_a_bt, 1.0) * (tf.add(e_a, 1.0) / e_a)
    return e_a_plus1

y = tf.placeholder(dtype=tf.float32)
t = tf.placeholder(dtype=tf.float32)


a = tf.Variable(0, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

f = f(a, b, t)
loss = tf.pow((y-f), 2)

train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss=loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    T1 = time.time()
    for i in range(1000):
        feed = {y:1, t:0}
        sess.run(train_op, feed)
        feed = {y:1, t:-1}
        sess.run(train_op, feed)
        feed = {y:1, t:-2}
        sess.run(train_op, feed)
        feed = {y:1, t:21}
        sess.run(train_op, feed)
        feed = {y:1, t:21}
        sess.run(train_op, feed)
        feed = {y:0.0, t:52}
        sess.run(train_op, feed)
        feed = {y:0.0, t:53}
        sess.run(train_op, feed)
    T2 = time.time()

    print(sess.run([a, b]), "," , "{:.6f}s".format(T2-T1))



