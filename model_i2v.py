"""

item2vec 训练文件
注意 读取的是 yoochoose.common 里面的 WINDOW_3_TRAINING_FILE 文件作为训练数据

"""

import tensorflow as tf


class item2vec(object):

    def __init__(self, hidden_units, n_sampled, item_size, learning_rate=0.001):
        self.item_size = item_size
        self.hidden_units = hidden_units
        self.n_sampled = n_sampled
        self.global_step = tf.Variable(0,trainable=False)
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
        with tf.name_scope('lables'):
            self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

        self.keep_prob = tf.placeholder(tf.float32)
        self.calculate_cost()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost,global_step=self.global_step)
        self.saver = tf.train.Saver(var_list={v.op.name: v for v in [self.embeding_matrix,self.softmax_b,self.softmax_w]},
                               max_to_keep=100)

    def calculate_cost(self):
        self.embeding_matrix = tf.Variable(tf.random_uniform([self.item_size, self.hidden_units], -1, 1),
                                           name='embedding_matrix')


        self.embed = tf.nn.embedding_lookup(self.embeding_matrix, self.inputs)

        # 添加 dropout层
        self.embed_dropout = tf.nn.dropout(self.embed, keep_prob=self.keep_prob)

        self.softmax_w = tf.Variable(tf.truncated_normal([self.item_size, self.hidden_units], stddev=0.1),
                                     name='softmax_w')

        self.softmax_b = tf.Variable(tf.zeros(self.item_size), name='softmax_b')

        self.cost = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, self.labels,
                                                              self.embed_dropout, self.n_sampled, self.item_size))
        tf.summary.scalar(name="cost",tensor=self.cost)
