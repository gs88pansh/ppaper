import tensorflow as tf
import numpy as np
import time
import pandas as pd
from _operator import itemgetter
from math import sqrt
import random
from Utils import *

class SCIV_kNN_TF_MODEL(object):
    def __init__(self, sess, item_size, restore_dir, restore_model, saver_dir,
                 n_session_sample=100, learning_rate=0.01):
        self.sess = sess
        self.item_size = item_size
        self.restore_dir = restore_dir
        self.restore_model = restore_model
        self.saver_dir = saver_dir
        self.n_session_sample = n_session_sample
        self.learning_rate = learning_rate
        self.zero = tf.constant(0, dtype=tf.int32)
        self.one = tf.constant(1, dtype=tf.int32)

        # 指定 place_hoder
        self.placehoders()
        self.a = tf.Variable(tf.zeros([self.item_size, 1]), trainable=True, dtype=tf.float32, name='a')
        self.b = tf.Variable(tf.zeros([self.item_size, 1]), trainable=True, dtype=tf.float32, name='b')

        # 损失函数
        # self.ce = self.CE(self.p(self.score()), self.MASK)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.score(),labels=self.MASK)
        self.a_g, self.b_g = tf.gradients(self.loss, [self.a, self.b] )

        self.top20 = tf.nn.top_k(self.score(), 20).indices
        # 定义训练过程
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # 定义saver
        self.saver = tf.train.Saver(var_list={v.op.name: v for v in tf.trainable_variables()}, max_to_keep=100)
        self.sess.run(tf.global_variables_initializer())

    def restoreModel(self, restorepath):
        self.saver.restore(self.sess, restorepath)

    def restoreModel(self, restorepath):
        self.saver.restore(self.sess, restorepath)

    def restoreModel(self, restorepath):
        self.saver.restore(self.sess, restorepath)

    def save_models(self, step):
        save_path = self.saver_dir + "/A{}".format(step)
        self.saver.save(self.sess,save_path=save_path)

    def getSparseTensor(self, indices):
        stv = tf.SparseTensorValue(indices=indices, values=np.ones([len(indices)]),
                               dense_shape=[self.item_size, self.n_session_sample])

    def placehoders(self):
        self.indices = tf.placeholder(tf.int64)
        self.values = tf.placeholder(tf.int32)
        self.shape = tf.placeholder(tf.int64)
        self.Ts = tf.placeholder(tf.float32, shape=[self.n_session_sample], name='Ts') # (n_session_sample)
        self.SIMs = tf.placeholder(tf.float32, shape=[self.n_session_sample], name='SIMs') # (n_session_sample)
        # self.IS_INs = tf.sparse_placeholder(tf.float32, name='IS_INs') # (item_size * n_session_sample)
        self.MASK = tf.placeholder(tf.float32, shape=[self.item_size], name="MASK") # (item_size)
        self.IS_INs = tf.SparseTensor(self.indices, self.values, self.shape)
    # 计算时间因子
    def f(self, a, b, t):
        '''
        计算 (exp(a)+1)/exp(a) * exp(a+bt) / (exp(a+bt)+1)
        :param a: tensor
        :param b: tensor
        :param t: tensor
        :return: tensor
        '''
        e_a = tf.exp(a) # exp(a)
        e_a_bt = tf.exp(
            tf.subtract(a,
                tf.multiply(
                    b,
                    t))) # exp(a-bt)
        e_a_plus1 = (tf.add(e_a, 1.0)/e_a) * (e_a_bt) / tf.add(e_a_bt, 1.0)
        return e_a_plus1
    # 计算评分
    # 计算概率
    def p(self, scores):
        e_scores = tf.exp(scores)
        sum = tf.reduce_sum(e_scores, 0)
        p = tf.divide(e_scores, sum)
        return p

    def CE(self, p, mask):
        return tf.log(tf.clip_by_value(tf.reduce_sum(tf.multiply(p, mask)),1e-8,1))

    def cond(self, i, indices, zero, one, scores):
            tf_shape = tf.shape(indices)
            length = tf_shape[zero]
            return tf.less(i, length)

    def body(self, i, indices, zero, one, scores):
        # x0 item_id
        # x1 sessionID
        x0 = indices[i][zero]
        x1 = indices[i][one]
        b_x0 = tf.nn.embedding_lookup(self.b, x0)
        t_x1 = tf.nn.embedding_lookup(self.Ts, x1)
        a_x0 = tf.nn.embedding_lookup(self.a, x0)
        mf = self.f(a_x0, b_x0, t_x1)
        mf_onehot = tf.one_hot(indices=x0, depth=self.item_size, on_value=1.0, off_value=0.0, axis=-1)
        scores = tf.add(scores, tf.multiply(mf_onehot, mf))
        scores.set_shape([self.item_size])
        i = tf.add(i, one)
        return i, indices, zero, one, scores

    def score(self):
        self.i = tf.constant(0, dtype=tf.int32)
        self.scores = tf.constant(np.zeros([self.item_size]), dtype=tf.float32)
        i, indices, zero, one, scores = tf.while_loop(cond=self.cond, body=self.body,
            loop_vars=[self.i, self.IS_INs.indices, self.zero, self.one, self.scores])
        return scores


def test_whileLoopTrain():
    Ts = np.ones(500)
    MASK = np.zeros(40000)
    MASK[1:10] = 1
    SIMs = np.ones(500)

    with tf.Session() as sess:
        model = SCIV_kNN_TF_MODEL(sess, 40000, n_session_sample=500)
        feed = {model.Ts: Ts, model.MASK: MASK, model.SIMs: SIMs,
                model.indices: [[0,0],[0,1],[1,0],[1,1]],
                model.values: np.ones([4]),
                model.shape: [40000, 500]}
        for i in range(10):
            T1 = time.time()
            _ = sess.run([model.train_op], feed_dict=feed)
            T2 = time.time()
            print(T2 - T1)
        a = sess.run(model.a)
        b = sess.run(model.b)
        print(a)
        print(b)







class SCIVKNN :

    def getDataSet(self, path):
        with open(path) as file_reader:
            itemsl = []
            for line in file_reader.readlines():
                strs = line.split("\n")[0].split(",")
                strs = strs[0:]
                index_arr = [int(v) for v in strs]
                index_arr[0] = float(strs[0])
                if (len(index_arr) >= 2):
                    itemsl.append(index_arr)
            itemsl.sort(key=lambda x:x[1])
            return itemsl

    def __init__(self, k, item_size, model, sample_size=1000, sampling='recent', similarity='jaccard', remind=False, pop_boost=0,
                 extend=False, normalize=True, session_key='SessionId', item_key='ItemId', time_key='Time'):

        self.k = k # session近邻个数，默认 100
        self.item_size = item_size
        self.sample_size = sample_size # 一步近邻，第一步近邻个数，那些有当前session里面的物品的session
        self.sampling = sampling # 抽样函数，一般取最近的，或者随机
        self.similarity = similarity # 相似函数的定义
        self.remind = remind # 最后的物品，最近点击的物品作为提醒，提高最近点击的物品的评分
        self.pop_boost = pop_boost # 提高受欢迎的物品的因子
        self.extend = extend # 是否添加已经评估过的session
        self.normalize = normalize # 是否归一化
        self.session_key = session_key # SessionId
        self.item_key = item_key # ItemId
        self.time_key = time_key # Time

        # cached
        self.item_session_map = dict() # itemID : sessionID_set
        self.session_item_map = dict() # sessionID : itemID_set
        self.session_time_map = dict() # sessionID : timestamp

        # updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        self.model = model
        self.sim_time = 0

        #

    def getT(self, days):
        return days // 10 * 10.0

    def fit(self, train, is_training=True, is_zero=False, restore_dir_model="",items=None):
        # train [time, itemID1, itemID2, ...]
        # 严格按照时间顺序的 session

        model.saver_dir += "sampling_{}".format(self.sampling)
        T1 = time.time()
        T2 = time.time()
        if not is_training:
            if not is_zero:
                self.model.restoreModel(restore_dir_model)
            a_, b_ = self.model.sess.run([self.model.a, self.model.b])
            print(a_,"\n",b_)
        for i in range(len(train)):
            if i >= 10000 and i % 10000 == 0:
                T2 = time.time()
                print("{}/{}个session训练时间：{}".format(i, len(train), T2-T1))
                T1 = time.time()
                if (is_training):
                    if i > len(train) - 20000:
                        self.model.save_models(i)
            now_session = train[i][1:]
            if len(now_session) == 1:
                continue
            session_time = train[i][0]
            session_id = i

            if is_training and i > len(train) - 20000:
                judgeDirExistAndDelCreate(model.saver_dir)
                MASK = np.zeros([self.item_size])
                MASK[now_session] = 1 / len(now_session)
                # 根据当前输入的 item_id 迅速选取的 k 个最相似的 session
                before_session = set()
                for itemID in now_session[:-1]:
                    indices = []
                    Ts = np.zeros([self.k])
                    SIMs = np.zeros([self.k])
                    before_session.add(itemID)
                    # neibors [(sid, sim),...]
                    neibors = self.find_neighbors(before_session, itemID, session_id)
                    cnt = 0
                    session_index_map = dict()
                    for neibor in neibors:
                        SIMs[cnt] = neibor[1]
                        Ts[cnt] = self.getT(self.session_time_map.get(neibor[0]) - session_time) / 86400.0 # 天数
                        for nItemID in self.session_item_map.get(neibor[0]):
                            indices.append([nItemID, cnt])
                        cnt += 1
                    feed = {model.Ts: Ts, model.MASK: MASK, model.SIMs: SIMs,
                            model.indices: indices,
                            model.values: np.ones(len(indices)),
                            model.shape: [model.item_size, model.n_session_sample]}
                    model.sess.run(model.train_op, feed)

            self.addTrainData(train[i], session_id)
        if (is_training):
            self.model.save_models("FFFFFF")

    def addTrainData(self, s, sID):
        now_session = s[1:]
        session_time = s[0]
        session_id = sID
        session_items = set(now_session)
        self.session_item_map.update({session_id: session_items})
        self.session_time_map.update({session_id: session_time})
        for itemID in session_items:
            itemSessionSet = set()
            if self.item_session_map.get(itemID) is None:
                self.item_session_map.update({itemID: itemSessionSet})
            else:
                itemSessionSet = self.item_session_map.get(itemID)
            itemSessionSet.add(session_id)

    def test(self, test):
        startIndex = 1000000
        yes = 0
        mrr = 0
        total = 0
        T1 = time.time()
        T2 = time.time()
        for i in range(len(test)):
            if i % 100 == 0 and i >= 100:
                T2 = time.time()
                print("时间：{} 上一次间隔：{:.0f}s  {}/{} recall@20:{:.5f} mrr@20:{:.5f}".format(dateNow(), T2-T1 , i, len(test), yes / total, mrr / total))
                T1 = time.time()
            session_id = startIndex + i
            session_time = test[i][0]
            items = test[i][1:]
            if len(items) <= 1:
                continue

            before_session = set()
            self.relevant_sessions = set()

            top20 = set()
            top20List = []
            total -= 1
            for itemID in items:
                total += 1
                if itemID in top20:
                    yes += 1
                    for index in range(len(top20List)):
                        if top20List[index] == itemID:
                            mrr += 1/(index+1)
                            break

                before_session.add(itemID)
                neibors = self.find_neighbors(before_session, itemID, session_id)
                Ts = np.zeros([self.k])
                SIMs = np.zeros([self.k])
                indices = []
                cnt = 0
                session_index_map = dict()
                for neibor in neibors:
                    SIMs[cnt] = neibor[1]
                    Ts[cnt] = (self.session_time_map.get(neibor[0]) - session_time) / (120 * 86400.0)  # 天数
                    for nItemID in self.session_item_map.get(neibor[0]):
                        indices.append([nItemID, cnt])
                    cnt += 1

                feed = {model.Ts: Ts, model.SIMs: SIMs,
                        model.indices: indices,
                        model.values: np.ones(len(indices)),
                        model.shape: [model.item_size, model.n_session_sample]}
                top20List = model.sess.run(model.top20, feed)
                top20 = set(top20List)
            self.addTrainData(test[i], i)



    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            点击事件的 session_id
        input_item_id : int or string
            点击事件的 item_id
        predict_for_item_ids : 1D array
            被预测的item_ids

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''

        #         gc.collect()
        #         process = psutil.Process(os.getpid())
        #         print( 'cknn.predict_next: ', process.memory_info().rss, ' memory used')

        if (self.session != session_id):  # new session
            # 实时扩展数据集
            if (self.extend):
                item_set = set(self.session_items)
                self.session_item_map[self.session] = item_set
                for item in item_set:
                    map_is = self.item_session_map.get(item)
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item: map_is})
                    map_is.add(self.session)

                ts = time.time()
                self.session_time_map.update({self.session: ts})

            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()

        self.session_items.append(input_item_id)

        if skip:
            return

        # 根据当前session里面的items来寻找相应的近邻session
        # neighbors : [(sid, sim),...]
        neighbors = self.find_neighbors(set(self.session_items), input_item_id, session_id)
        # 取完近邻后，对物品进行评分
        # scores : {itemID : score}
        scores = self.score_items(neighbors)

        # add some reminders
        if self.remind:

            reminderScore = 5
            takeLastN = 3

            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1
                # reminderScore = reminderScore + (cnt/100)

                oldScore = scores.get(elem)
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                # print 'old score ', oldScore
                # update the score and add a small number for the position
                newScore = (newScore * reminderScore) + (cnt / 100)
                scores.update({elem: newScore})

        # push popular ones
        if self.pop_boost > 0:
            # {itemID : 流行度（最大为1）}
            pop = self.item_pop(neighbors)
            # Iterate over the item neighbors
            # print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key: (scores[key] + (self.pop_boost * item_pop))})

        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, list(scores.keys()))

        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)
        if self.normalize:
            series = series / series.max()
        return series

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)
        '''
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session(session)
            for item in items:

                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})

                if (result.get(item) > max_pop):
                    max_pop = result.get(item)

        for key in result:
            result.update({key: (result[key] / max_pop)})

        return result

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second)
        res = intersection / union

        self.sim_time += (time.clock() - sc)

        return res

    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result

    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / (la + lb - li)

        return result

    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        a = len(first & second)
        b = len(first)
        c = len(second)

        result = (2 * a) / ((2 * a) + b + c)

        return result

    def items_for_session(self, session):
        # 获得session里面的item
        return self.session_item_map.get(session)

    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        Parameters
        --------
        item: Id of the item session
        Returns
        --------
        out : set
        '''
        result = self.item_session_map.get(item_id)
        if result is None:
            return set()
        return self.item_session_map.get(item_id)

    def most_recent_sessions(self, sessions, number):
        # 按照时间最近抽样
        sample = set()
        tuples = list()
        for session in sessions:
            time = self.session_time_map.get(session)
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        # print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add(element[0])
        # print 'returning sample of size ', len(sample)
        return sample

    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        # 函数找到可能的邻居子集

        # 并集，这里每次都求一次并集，因此每一次只取最后一个就可以了
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(input_item_id)

        if self.sample_size == 0:  # use all session as possible neighbors
            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions
        else:  # sample some sessions
            # 预先抽样
            # self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(input_item_id)
            if len(self.relevant_sessions) > self.sample_size:

                if self.sampling == 'recent':
                    sample = self.most_recent_sessions(self.relevant_sessions, self.sample_size)
                elif self.sampling == 'random':
                    sample = random.sample(self.relevant_sessions, self.sample_size)
                else:
                    sample = self.relevant_sessions[:self.sample_size]

                return sample
            else:
                return self.relevant_sessions

    def calc_similarity(self, session_items, sessions):
        # 计算session相似度
        # 返回 [(sessionID, similarity),...]

        # print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first
            session_items_test = self.items_for_session(session)
            # getattr 根据后面的找前面的属性，换句话说，找到对应的函数
            similarity = getattr(self, self.similarity)(session_items_test, session_items)
            if similarity > 0:
                neighbors.append((session, similarity))
        return neighbors

    # -----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity)
    # -----------------
    def find_neighbors(self, session_items, input_item_id, session_id):
        # 寻找可能的 session
        possible_neighbors = self.possible_neighbor_sessions(session_items, input_item_id, session_id)
        # 计算每个可能的 session 与当前 session 的相似度
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors)
        # possible_neighbors：  [(sid1, sim), (sid2,sim), ...]
        # 对相似度进行排序，从高到低
        possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])
        # 去相似度最高的 k 个session
        possible_neighbors = possible_neighbors[:self.k]
        return possible_neighbors

    def score_items(self, neighbors):
        '''
        计算每个物品的评分
        out : dict
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session(session[0])
            for item in items:
                old_score = scores.get(item)
                new_score = session[1]

                if old_score is None:
                    scores.update({item: new_score})
                else:
                    new_score = old_score + new_score
                    scores.update({item: new_score})

        return scores


if __name__ == "__main__":
    # test_whileLoopTrain()

    dataSetName = "diginetica"
    with tf.Session() as sess:
        learning_rate = 0.002
        item_size = 39187
        k = 100
        restore_dir = ""
        restore_model = ""
        saver_dir = "./{}/model/scivknn/k{}le{}".format(dataSetName, k, learning_rate)


        model = SCIV_kNN_TF_MODEL(sess, item_size, restore_dir=restore_dir, restore_model=restore_model,learning_rate=learning_rate,
                                  saver_dir=saver_dir, n_session_sample=k)

        scivknn = SCIVKNN(k, item_size, model, sampling="random")
        train_seq_arr = scivknn.getDataSet("./{}/data/preprocessed/train-plain-seq.txt".format(dataSetName))
        test_seq_arr = scivknn.getDataSet("./{}/data/preprocessed/test-seq.txt".format(dataSetName))

        print("训练数据加载完毕...")

        scivknn.fit(train_seq_arr, True)
        # scivknn.fit(train_seq_arr, False, True)
        # scivknn.fit(train_seq_arr, False, "./{}/model/scivknn/k100le0.001/AFFFFFF".format(dataSetName))


        scivknn.test(test_seq_arr)




