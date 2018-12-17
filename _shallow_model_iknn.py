import numpy as np
import pandas as pd


class ItemKNN:
    '''
    ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time')
    物品kNN 实现计算好不同物品及之间的相似度
    物品之间的相似度定义由下式计算得出
    .. math::
        s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}
    Parameters
    --------
    n_sims : int
        Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    lmbd : float
        Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
    alpha : float
        Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')

    '''

    def __init__(self, n_sims=100, lmbd=20, alpha=0.5, session_key='SessionId', item_key='ItemId', time_key='Time'):
        self.n_sims = n_sims
        self.lmbd = lmbd  # 如果物品数过小，就用这个数代替之
        self.alpha = alpha
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key

    def fit(self, data):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        data.set_index(np.arange(len(data)), inplace=True)
        itemids = data[self.item_key].unique()
        # item个数
        n_items = len(itemids)
        # 将 ID 转化为 新的 ID
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': np.arange(len(itemids))}),
                        on=self.item_key, how='inner')
        sessionids = data[self.session_key].unique()
        data = pd.merge(data, pd.DataFrame({self.session_key: sessionids, 'SessionIdx': np.arange(len(sessionids))}),
                        on=self.session_key, how='inner')

        # 每个 session 的长度差搞出来
        supp = data.groupby('SessionIdx').size()
        session_offsets = np.zeros(len(supp) + 1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum()  # [0,3,7...] s0长度3，s1长度4
        # 按照 SessionIdx time 排好序的
        index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values

        supp = data.groupby('ItemIdx').size()
        item_offsets = np.zeros(n_items + 1, dtype=np.int32)
        item_offsets[1:] = supp.cumsum()  # [0,3,7...] i0长度3，i1长度4
        # 按照 ItemIdx time 排好序的
        index_by_items = data.sort_values(['ItemIdx', self.time_key]).index.values

        # key: ItemIdx, value:
        self.sims = dict()

        # item的个数
        for i in range(n_items):
            iarray = np.zeros(n_items)
            start = item_offsets[i]
            end = item_offsets[i + 1]
            # index_by_items[start:end] 取出所有的包含物品 i 的 clicks
            for e in index_by_items[start:end]:
                # e 对应着每一个包含物品 i 的 click

                # uidx 包含物品 i 的sessionID
                uidx = data.SessionIdx.values[e]
                ustart = session_offsets[uidx]
                uend = session_offsets[uidx + 1]

                # sessionID 中包含的 clicks
                # user_events 是一个 numpy
                user_events = index_by_sessions[ustart:uend]

                # 物品 i 与其他物品有公共的session的个数
                iarray[data.ItemIdx.values[user_events]] += 1

            iarray[i] = 0
            # supp[i] 物品i的个数
            norm = np.power((supp[i] + self.lmbd), self.alpha) \
                   * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            # argsort 返回排好序的 数组index,注意，这里从 -1 开始的，因此很好
            indices = np.argsort(iarray)[-1:-1 - self.n_sims:-1]
            self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])

    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''
        preds = np.zeros(len(predict_for_item_ids))

        # series index:itemID,data:评分
        sim_list = self.sims[input_item_id]

        # mask 里面存的是 [true, false, true, false, ...]
        mask = np.in1d(predict_for_item_ids, sim_list.index)

        # 展开到更高维度上
        preds[mask] = sim_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)
