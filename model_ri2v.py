import tensorflow as tf


class RI2V(object):

    def __init__(self,sess,args,is_training=True):
        self.embedding_size, self.hidden_size, self.item_size, \
        self.batch_size, self.n_sampled, self.learning_rate, self.joint_train = \
        args.embedding_size, args.hidden_size, args.item_size, \
        args.batch_size, args.n_sampled, args.learning_rate, args.joint_train
        self.saver_dir = args.saver_dir
        self.restore_dir = args.restore_dir
        self.restore_model = args.restore_model
        self.restore_i2v_dir = args.restore_i2v_dir
        self.restore_i2v_model = args.restore_i2v_model
        self.restore_ri2v_dir = args.restore_ri2v_dir
        self.restore_ri2v_model = args.restore_ri2v_model
        self.sess = sess
        self.summary_path = args.summary_path
        self.is_training = is_training

        self.placehoders()
        embed = self.embedding_layer(name="embedding_layer")
        outputs, final_state = self.rnn_layer(embed,"rnn_layer",time_major=False)
        rnn_output_X, final_rnn_output_Y = self.rnn_out_layer(outputs,final_state,name="rnn_out_layer")

        self.softmax_w = tf.Variable(tf.zeros([self.item_size,self.embedding_size,]),trainable=self.joint_train,dtype=tf.float32,name='softmax_w')
        self.softmax_b = tf.Variable(tf.zeros(self.item_size),trainable=self.joint_train,dtype=tf.float32,name='softmax_b')

        logitX,final_logitY = self.logit_layer(rnn_output_X,final_rnn_output_Y,self.softmax_w,self.softmax_b,name="logit_layer")

        self.top_1,self.top_5,self.top_10,self.top_20,self.total,\
        self.last_1,self.last_5,self.last_10,self.last_20,self.last_total \
            = self._predict(self.softmax_w,self.softmax_b,logitX,final_logitY,name="predict")

        self.recall_at_20 = self.last_20

        self.last_mrr_at_20 = self.MRR_at_k(final_logitY,self.last_labels,20)
        self.last_mrr_at_10 = self.MRR_at_k(final_logitY,self.last_labels,10)
        self.last_mrr_at_5 = self.MRR_at_k(final_logitY,self.last_labels,5)
        self.last_mrr_at_1 = self.MRR_at_k(final_logitY,self.last_labels,1)

        self.mrr_at_20 = self.MRR_at_k(logitX,tf.reshape(self.labels,[-1]))
        self.mrr_at_10 = self.MRR_at_k(logitX,tf.reshape(self.labels,[-1]),10)
        self.mrr_at_5 = self.MRR_at_k(logitX,tf.reshape(self.labels,[-1]),5)
        self.mrr_at_1 = self.MRR_at_k(logitX,tf.reshape(self.labels,[-1]),1)

        self.loss_neg_sample = self._neg_sample(self.softmax_w,self.softmax_b,rnn_output_X,name="loss_neg_sample")

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_neg_sample)
        self.merge_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.summary_path, tf.get_default_graph())


        self.sess.run(tf.global_variables_initializer())

        if not self.is_training :
            if not self.joint_train:
                restore_i2v = tf.train.Saver(var_list={"embedding_matrix": self.embedding_matrix, "softmax_w": self.softmax_w, "softmax_b": self.softmax_b, }, name="restore",max_to_keep=100)
                restore_i2v.restore(sess, save_path=self.restore_i2v_dir + "/" + self.restore_i2v_model)
                restore_ri2v = tf.train.Saver(var_list={v.op.name: v for v in tf.trainable_variables()}, max_to_keep=100)
                restore_ri2v.restore(sess, save_path=self.restore_ri2v_dir + "/" + self.restore_ri2v_model)
                vlist = [self.softmax_w, self.softmax_b, self.embedding_matrix]
                vlist.extend(tf.trainable_variables())
                self.saver = tf.train.Saver(var_list={v.op.name: v for v in vlist}, max_to_keep=100)
            else:
                self.saver = tf.train.Saver(var_list={v.op.name: v for v in tf.trainable_variables()}, max_to_keep=100)
                self.saver.restore(sess, save_path=self.restore_dir + "/" + self.restore_model)

        else:
            if self.joint_train:
                self.saver = tf.train.Saver(var_list={v.op.name : v for v in tf.trainable_variables()},max_to_keep=100)
                self.saver.restore(sess,save_path=self.restore_dir +"/"+ self.restore_model)
            else:
                vlist = [self.softmax_w,self.softmax_b,self.embedding_matrix]
                vlist.extend(tf.trainable_variables())
                self.saver =  tf.train.Saver(var_list={v.op.name : v for v in vlist},max_to_keep=100)
                restore = tf.train.Saver(
                    var_list={"embedding_matrix": self.embedding_matrix, "softmax_w": self.softmax_w, "softmax_b": self.softmax_b, },
                    name="restore")
                restore.restore(sess,save_path=self.restore_i2v_dir +"/"+ self.restore_i2v_model)

    def log(self,feed,j):
        su = self.sess.run(self.merge_op,feed_dict=feed)
        self.summary_writer.add_summary(su,j)
    def save(self,step):
        save_path = self.saver_dir + "/A{}".format(step)
        self.saver.save(self.sess,save_path=save_path)
    def placehoders(self):
        with tf.name_scope('inputs'):
            # batch_size * step_size
            self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
        with tf.name_scope('lables'):
            self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        with tf.name_scope('keep_prob'):
            self.keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('last_labels'):
            self.last_labels = tf.placeholder(tf.int32,shape=[None],name="last_labels")
    def embedding_layer(self,name="embedding_layer"):
        with tf.name_scope(name):
            self.embedding_matrix = tf.Variable(tf.zeros([self.item_size,self.embedding_size,]),trainable=self.joint_train,dtype=tf.float32,name='embedding_matrix')
            tf.summary.histogram("embedding_matrix",self.embedding_matrix)
            # embed: 'batch_size * step_size * embeding_size'
            embed = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)
            return embed
    def rnn_layer(self,X,name="rnn_layer",time_major=False,):
        """
        :param X:
        :param name:
        :param time_major:
        :return:
            outputs: shape[batch_size,rnn_steps,rnn_states]
            final_states: shape[batch_size,rnn_states]
        """
        cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_size)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, time_major=time_major)
        return outputs, final_state
    def rnn_out_layer(self,X,final_X,name="rnn_out_layer"):
        with tf.name_scope(name):
            # X: shape [batch_size*rnn_steps,rnn_states]
            # Y: shape [batch_size,rnn_states]
            X = tf.reshape(X, [-1, self.hidden_size])
            rnn_output_W = tf.Variable(tf.random_uniform([self.hidden_size, self.embedding_size], -1, 1), name='rnn_output_W')
            rnn_output_b = tf.Variable(tf.zeros(self.embedding_size), name='rnn_output_b')
            rnn_output_X = tf.nn.dropout(tf.matmul(X,rnn_output_W) + rnn_output_b,keep_prob=self.keep_prob)
            rnn_output_Y = tf.nn.dropout(tf.matmul(final_X,rnn_output_W) + rnn_output_b,keep_prob=self.keep_prob)
            return rnn_output_X,rnn_output_Y
    def logit_layer(self,X,Y,softmax_w,softmax_b,name="logit_layer"):
        with tf.name_scope(name):
            logitX = tf.matmul(X, tf.transpose(softmax_w)) + softmax_b
            logitY = tf.matmul(Y, tf.transpose(softmax_w)) + softmax_b
            return logitX,logitY
    def _neg_sample(self,softmax_w,softmax_b,logitX,name="loss_neg_sample"):
        with tf.name_scope(name):
            labels_plat = tf.reshape(self.labels, [-1, 1])
            loss_neg_sample = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels_plat,logitX, self.n_sampled, self.item_size))
            tf.summary.scalar("loss_neg_sample",loss_neg_sample)
            return loss_neg_sample
    def _predict(self,softmax_w,softmax_b,logitX,final_logitY,name="predict"):
        with tf.name_scope(name):
            labels_plat = tf.reshape(self.labels,[-1])
            top_1 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logitX, labels_plat, 1), tf.float32))
            top_5 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logitX, labels_plat, 5), tf.float32))
            top_10 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logitX, labels_plat, 10), tf.float32))
            top_20 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logitX, labels_plat, 20), tf.float32))
            total = tf.size(labels_plat)

            last_1 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(final_logitY, self.last_labels, 1), tf.float32))
            last_5 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(final_logitY, self.last_labels, 5), tf.float32))
            last_10 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(final_logitY, self.last_labels, 10), tf.float32))
            last_20 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(final_logitY, self.last_labels, 20), tf.float32))
            last_total = tf.size(self.last_labels)

            return top_1,top_5,top_10,top_20,total,last_1,last_5,last_10,last_20,last_total
    def MRR_at_k(self,score_tensor, target_tensor, k = 20):
        r = tf.nn.top_k(score_tensor, k).indices
        r_T = tf.transpose(r, [1, 0])
        equal = tf.transpose(tf.equal(r_T, target_tensor), [1, 0])
        where = tf.where(equal)
        slice = tf.cast(tf.slice(where, [0, 1], [tf.shape(where)[0], 1]), tf.float32)
        mrr = tf.reduce_sum(1.0 / (slice + 1)) / tf.cast(tf.shape(score_tensor)[0], tf.float32)
        return mrr
