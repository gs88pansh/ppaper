from _operator import itemgetter
from math import sqrt
import random
import time

import numpy as np
import pandas as pd


class ContextKNN:
    '''
    ContextKNN( k, sample_size=500, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, session_key = 'SessionId', item_key= 'ItemId')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__(self, k, sample_size=1000, sampling='recent', similarity='jaccard', remind=False, pop_boost=0,
                 extend=False, normalize=True, session_key='SessionId', item_key='ItemId', time_key='Time'):

        self.k = k # session近邻个数，默认 100
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

        # updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()

        self.sim_time = 0

    def fit(self, train, items=None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''

        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)
        index_time = train.columns.get_loc(self.time_key)

        session = -1
        session_items = set()
        time = -1
        # cnt = 0

        # iterrows(): 将DataFrame迭代为(insex, Series)对。
        # itertuples(): 将DataFrame迭代为元祖。Pandas(index=0, col1=.., col2=.., ...)
        # iteritems(): 将DataFrame迭代为(列名, Series)对
        # 填充 session_item_map 和 item_session_map
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session: session_items})
                    # cache the last time stamp of the session
                    self.session_time.update({session: time})
                session = row[index_session]
                session_items = set()
            time = row[index_time]
            session_items.add(row[index_item])

            # cache sessions involving an item
            map_is = self.item_session_map.get(row[index_item])
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item]: map_is})
            map_is.add(row[index_session])

        # Add the last tuple
        self.session_item_map.update({session: session_items})
        self.session_time.update({session: time})

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
                self.session_time.update({self.session: ts})

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
        return self.item_session_map.get(item_id)

    def most_recent_sessions(self, sessions, number):
        # 按照时间最近抽样
        sample = set()
        tuples = list()
        for session in sessions:
            time = self.session_time.get(session)
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
