import tensorflow as tf
import numpy as np
import time

def test_tensorOperation():
    x = tf.Variable([1.0, 2.0], dtype=tf.float32)
    y = tf.Variable([3.0, 4.0], dtype=tf.float32)

    # element wise operation
    m = tf.add(x, y) # 加法运算
    m = tf.subtract(x, y) # 减法运算
    m = tf.multiply(x, y) # 乘法运算
    m = tf.divide(x, y) # 除法运算
    m = tf.exp(x) # 指数运算
    m = tf.add(x, 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nm = sess.run(m)
        print(nm)

def test_notNormalTensor():
    x = tf.Variable([[3., 2,], [2.]], dtype=tf.float32)
    result = tf.add(x, 1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nm = sess.run(result)
        print(nm)

def test_notSameShape():
    T1 = time.time()
    npzeros = np.ones([40000, 400]) # 0.001s # 0.09
    T2 = time.time()
    print(T2 - T1)
    a = tf.Variable(npzeros, dtype=tf.float64) # 0.31s
    e = tf.reduce_sum(a, axis=1) # 0.017
    e = tf.multiply(a, 0) # 0.079
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        T1 = time.time()
        print(sess.run(e))
        T2 = time.time()
        print(T2 - T1)

def test_reshape():
    a = tf.Variable([1,1,1])
    b = tf.reshape(a, [3, -1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        T1 = time.time()
        print(sess.run(b))
        T2 = time.time()
        print(T2 - T1)

def test_where():
    # Make a tensor from a constant
    a = np.reshape(np.arange(24), (3, 4, 2))
    a_t = tf.constant(a)
    # Find indices where the tensor is not zero
    idx = tf.where(tf.not_equal(a_t, 0))
    # Make the sparse tensor
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape()
    # if tensor shape is dynamic
    sparse = tf.SparseTensor(idx, tf.gather_nd(a_t, idx), a_t.get_shape())
    # Make a dense tensor back from the sparse one, only to check result is correct
    dense = tf.sparse_tensor_to_dense(sparse)
    # Check result
    reduce_sum = tf.reduce_sum(tf.sparse_tensor_to_dense(sparse), 1)
    with tf.Session() as sess:
        b, rs = sess.run([dense, reduce_sum])
    print(np.all(a == b))
    print(rs)

# 稀疏矢量大测试
def test_sparseMatrix():
    pass

def test_randomRead():
    # 注意，tensor的每一步操作都是一个 graph，不是容器意义上的读取
    data = tf.constant([[1, 2, 3], [4, 5, 6]])
    posi = tf.convert_to_tensor([0, 2])
    shape = data.get_shape()
    re = data[posi[0]][posi[1]]
    with tf.Session() as sess:
        print(type(re))
        print(type(shape)) # TensorShape
        print(sess.run(re))
        print(sess.run(data))

def test_sparseInputAndOperation():
    pass

def test_sparceReduce():
    # 注意，tensor的每一步操作都是一个 graph，不是容器意义上的读取
    data = tf.constant([[1,0,0], [0,0,0]])
    posi = tf.convert_to_tensor([0, 2])
    shape = data.get_shape()
    re = data[posi[0]][posi[1]]
    with tf.Session() as sess:
        print(type(re))
        print(type(shape)) # TensorShape
        print(sess.run(re))
        print(sess.run(data))

def test_oneHot():
    labels = 1
    # labels是shape=(4,)的张量。则返回的targets是shape=(len(labels), depth)张量。
    # 且这种情况下,axis=-1等价于axis=1
    targets = tf.one_hot(indices=labels, depth=5, on_value=1.0, off_value=0.0, axis=-1)
    with tf.Session() as sess:
        print(sess.run(targets))

def test_feedSparceMatrix():
    pass


def test_readModel():
    a = tf.Variable(tf.zeros([39187, 1]), trainable=True, dtype=tf.float32, name='a')
    b = tf.Variable(tf.zeros([39187, 1]), trainable=True, dtype=tf.float32, name='b')

    saver = tf.train.Saver({"a": a, "b": b}, max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, "../diginetica/model/scivknn/k100le0.01sampling_random/AFFFFFF")
        a_, b_ = sess.run([a,b])
        print(a_)
        for i in range(len(a_)) :
            if a_[i][0] > 0.7:
                print("[", str(a_[i][0]) + ',' + str(b_[i][0]), "],")

def test_readI2VModel():
    a = tf.Variable(tf.zeros([39187, 100]), trainable=True, dtype=tf.float32, name='a')
    # b = tf.Variable(tf.zeros([39187, 100]), trainable=True, dtype=tf.float32, name='b')

    saver = tf.train.Saver({"softmax_w": a}, max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "../diginetica/model/i2v/em100ba512sa256le0.0007/A20")
        a_ = sess.run([a])
        print(a_)

def test_genNewRandom():
    a = tf.placeholder(dtype=tf.float32, shape=[2,2])
    b = tf.add(a, tf.random_uniform([2,2],minval=-10, maxval=10))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            feed = {a:[[0,0],[0,0]]}
            a_ = sess.run([b],feed_dict=feed)
            print(a_)


if __name__ == "__main__":
    # test_tensorOperation()
    # test_notNormalTensor()
    # test_notSameShape()
    # test_reshape()
    # test_where()
    # test_randomRead()
    # test_oneHot()
    # test_readModel()
    # test_readI2VModel()
    test_genNewRandom()