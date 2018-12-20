import pandas as pd
import numpy as np
import datetime as dt
from Utils import *

import sys

def readData(dataSetName, raw_data_file):
    # 打开原始数据
    f = open(raw_data_file, "r")

    data_set = None
    if (dataSetName == "yoochoose"):
        data_set = pd.read_csv(f, sep=',', header=None, usecols=[0, 1, 2], dtype={0: np.int32, 1: str, 2: np.int64})
        data_set.columns = ['SessionId', 'TimeStr', 'OldItemId']
        data_set['Time'] = data_set.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
    elif (dataSetName == "diginetica"):
        data_set = pd.read_csv(f, sep=';', usecols=[0, 2, 4], dtype={0: np.int32, 2: np.int64, 4: str})
        data_set.columns = ['SessionId', 'OldItemId', 'TimeStr']
        data_set['Time'] = data_set.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp())
    del(data_set['TimeStr'])
    return data_set

def filterData(data_set, item_num_filter, session_num_max):
    session_lengths = data_set.groupby('SessionId').size()
    data_set = data_set[np.in1d(data_set.SessionId, session_lengths[session_lengths > 1].index)]


    # 在此删除 长度小于2并且大于 session_num_max 的记录
    session_lengths = data_set.groupby('SessionId').size()
    data_set = data_set[np.in1d(data_set.SessionId, session_lengths[session_lengths <= session_num_max].index)]

    # 删除 items 个数小于 item_num_filter 的记录
    item_supports = data_set.groupby('OldItemId').size()
    data_set = data_set[np.in1d(data_set.OldItemId, item_supports[item_supports >= item_num_filter].index)]

    data_set = data_set[np.in1d(data_set.SessionId, session_lengths[session_lengths >= 2].index)]
    return data_set

def train_test_last_n_days_split(data_set, test_days, last_n_days):
    tmax = data_set.Time.max()
    session_max_times = data_set.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-test_days*24*60*60].index
    session_test = session_max_times[session_max_times >= tmax-test_days*24*60*60].index
    train = data_set[np.in1d(data_set.SessionId, session_train)]

    train_max_times = train.groupby('SessionId').Time.max()
    session_last_n_days_train = train_max_times[session_max_times > tmax-test_days*24*60*60-last_n_days*24*60*60].index
    last_n_day = train[np.in1d(train.SessionId, session_last_n_days_train)]

    test = data_set[np.in1d(data_set.SessionId, session_test)]
    test = test[np.in1d(test.OldItemId, train.OldItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    return train, test, last_n_day

def newItemId(train_set, test_set, last_n_days, dataSetName):
    item_lengths = train_set.groupby("OldItemId").size().sort_values(ascending = False) #降序排列
    log(".\\{}\\data\\preprocess.log".format(dataSetName), "物品个数：{}".format(len(item_lengths)))
    map = {}
    for i in range(len(item_lengths)):
        map[item_lengths.index[i]] = i
    train_set["ItemId"] = train_set.OldItemId.apply(lambda x : map[x])
    test_set["ItemId"] = test_set.OldItemId.apply(lambda x : map[x])
    last_n_days["ItemId"] = last_n_days.OldItemId.apply(lambda x : map[x])
    del(train_set["OldItemId"])
    del(test_set["OldItemId"])
    return train_set, test_set, last_n_days

def save_sequence(train_set, test_set, last_n_days,
                  train_seq_path, test_seq_path, last_n_seq_path, train_plain_seq_path,
                  dataSetName):
    testKV = getKVFromDataFrame(test_set)
    _save_sequence(testKV, test_seq_path)
    lastnKV = getKVFromDataFrame(last_n_days)
    _save_sequence(lastnKV, last_n_seq_path)
    trainKV = getKVFromDataFrame(train_set)
    _save_sequence(trainKV, train_plain_seq_path)

    log(".\\{}\\data\\preprocess.log".format(dataSetName), "训练集click次数数：{} 个".format(len(train_set)))
    log(".\\{}\\data\\preprocess.log".format(dataSetName), "训练集session个数：{} 个".format(len(trainKV)))
    log(".\\{}\\data\\preprocess.log".format(dataSetName), "session平均长度：{} ".format(len(train_set) / len(trainKV)))

    train_maxdate = train_set.Time.max()
    train_mindate = train_set.Time.min()

    log(".\\{}\\data\\preprocess.log".format(dataSetName), "训练数据时间跨度：{} 天".format((train_maxdate-train_mindate)//(24*60*60)))
    train_time_over_max_min = train_maxdate - train_mindate
    sp_1_2 = train_time_over_max_min / 2 + train_mindate  # < *1
    sp_3_4 = train_time_over_max_min * 3 / 4 + train_mindate  # < *2
    sp_7_8 = train_time_over_max_min * 7 / 8 + train_mindate  # < *3
    sp_15_16 = train_time_over_max_min * 15 / 16 + train_mindate  # < *4
    # else *6
    train_1_2 = []
    train_3_4 = []
    train_7_8 = []
    train_15_16 = []
    train_16_16 = []
    for s in list(trainKV):
        cur = trainKV[s]
        if (cur[0] < sp_1_2):
            train_1_2.append(cur)
        elif (cur[0] < sp_3_4):
            train_3_4.append(cur)
        elif (cur[0] < sp_7_8):
            train_7_8.append(cur)
        elif (cur[0] < sp_15_16):
            train_15_16.append(cur)
        else:
            train_16_16.append(cur)
    finally_train_set = \
        [train_1_2,
         train_3_4, train_3_4,
         train_7_8, train_7_8, train_7_8,
         train_15_16, train_15_16, train_15_16, train_15_16,
         train_16_16, train_16_16, train_16_16, train_16_16, train_16_16, train_16_16]
    writeSessionSeqArrs(finally_train_set, train_seq_path)
    return trainKV, testKV, lastnKV

def writeSessionSeqArrs(dataArrArr, writePath):
    judgeFileExistAndDelCreateSlash(writePath)
    with open(writePath, "a+") as f:
        total = 0
        fLine = ""
        for arrarr in dataArrArr:
            for arr in arrarr:
                total += 1
                v = arr
                fLine += str(v[0])
                k = 0
                while k < len(v) - 1:
                    fLine += "," + str(v[k + 1])
                    k += 1
                fLine += "\n"
                if (total % 100000 == 0):
                    f.write(fLine)
                    fLine = ""
        f.write(fLine)

def getKVFromDataFrame(data):
    kv = {}
    for index, row in data.iterrows():
        sessionID = row["SessionId"]
        if sessionID in kv:
            kv[sessionID].append(int(row["ItemId"]))
        else:
            kv[sessionID] = []
            kv[sessionID].append(int(row["Time"]))
            kv[sessionID].append(int(row["ItemId"]))
    return kv

def _save_sequence(kv, path):
    judgeFileExistAndDelCreateSlash(path)
    with open(path,"a+") as f:
        total = 0
        fLine = ""
        for sessionId in kv.keys():
            total += 1
            v = kv[sessionId]
            fLine += str(v[0])
            k = 0
            while k < len(v)-1 :
                fLine += "," + str(v[k + 1])
                k+=1
            fLine += "\n"
            if (total % 100000 == 0):
                f.write(fLine)
                fLine = ""
        f.write(fLine)

def getItem2VecTrainingData(KV, window_size, dataSetName):
    '''
    构造一个获取batch的生成器
    '''
    train_x, train_y = [], []
    for k in KV.keys():
        idx_items = KV[k][1:]
        batch_size = len(idx_items)
        for i in range(len(idx_items)):
            batch_x = idx_items[i]
            batch_y = get_targets(idx_items, i, window_size)
            # 由于一个input word会对应多个output word，因此需要长度统一
            train_x.extend([batch_x] * len(batch_y))
            train_y.extend(batch_y)
    log(".\\{}\\data\\preprocess.log".format(dataSetName), "item2vec训练数据量：{}".format(len(train_x)))
    return train_x, train_y

def get_targets(words, idx, window_size=3):
    '''
    获得input word的上下文单词列表

    参数
    ---
    words: 单词列表
    idx: input word的索引号
    window_size: 窗口大小
    '''
    target_window = np.random.randint(1, window_size + 1)
    # 这里要考虑input word前面单词不够的情况
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    # print("target_window:",target_window,"start:",start_point,"end:",end_point)
    # output words(即窗口中的上下文单词)
    targets = set(words[start_point: idx] + words[idx + 1: end_point + 1])
    return list(targets)

def writeItem2VecTrainingData(path,x,y):
    judgeFileExistAndDelCreateSlash(path)
    with open(path, "a+") as f:
        i = 0
        lines = ""
        while i<len(x) :
            lines += str(x[i]) + "," + str(y[i]) + "\n"
            i += 1
            if i % 20000 == 0:
                f.write(lines)
                lines = ""
        f.write(lines)

def save_item2vec(KV, window_size, path, dataSetName):
    train_x, train_y = getItem2VecTrainingData(KV, window_size, dataSetName)
    writeItem2VecTrainingData(path, train_x, train_y)


def preprocess(dataSetName, raw_data_file, item_num_filter, session_num_max, num_last_days, window_size, test_days,
               train_seq_path = "/data/preprocessed/train-seq.txt",
               test_seq_path = "/data/preprocessed/test-seq.txt",
               last_n_seq_path = "/data/preprocessed/last-n-days.txt",
               i2v_path = "/data/preprocessed/i2v.txt",
               train_expand_path = "/data/preprocessed/train-expand.txt",
               test_expand_path="/data/preprocessed/test-expand.txt",
               train_plain_seq_path = "/data/preprocessed/train-plain-seq.txt"):

    train_seq_path = "./" + dataSetName + train_seq_path
    test_seq_path = "./" + dataSetName + test_seq_path
    last_n_seq_path = "./" + dataSetName + last_n_seq_path
    i2v_path = "./" + dataSetName + i2v_path
    train_expand_path = "./" + dataSetName + train_expand_path
    test_expand_path = "./" + dataSetName + test_expand_path
    train_plain_seq_path = "./" + dataSetName + train_plain_seq_path

    print("\n\t dataSetName: {} ".format(dataSetName) + ""
          "\n\t raw_data_file: {}".format(raw_data_file) + ""
          "\n\t item_num_filter: {}".format(item_num_filter) + ""
          "\n\t session_num_max: {}".format(session_num_max) + ""
          "\n\t num_last_days: {}".format(num_last_days) + ""
          "\n\t window_size: {}".format(window_size) + ""
          "\n\t test_days: {}".format(test_days) + ""
          "\n\t train_seq_path: {}".format(train_seq_path) + ""
          "\n\t test_seq_path: {}".format(test_seq_path) + ""
          "\n\t last_n_seq_path: {}".format(last_n_seq_path) + ""
          "\n\t i2v_path: {}".format(i2v_path) + ""
          "\n\t train_expand_path: {}".format(train_expand_path) + ""
          "\n\t test_expand_path: {}".format(test_expand_path) + ""
          "")

    log(".\\{}\\data\\preprocess.log".format(dataSetName), "preprocess*************{}************{}".format(dataSetName, dt.datetime.now()))
    data_set = readData(dataSetName, raw_data_file) # DataFrame

    #print(data_set)
    data_set = filterData(data_set, item_num_filter, session_num_max)
    train_set, test_set, last_n_days = train_test_last_n_days_split(data_set, test_days, num_last_days)

    #print(train_set)
    train_set, test_set, last_n_days = newItemId(train_set, test_set,last_n_days, dataSetName)
    train_set.to_csv(train_expand_path, sep='\t', index=False)
    test_set.to_csv(test_expand_path, sep='\t', index=False)
    #print(train_set)
    trainKV, testKV, lastKV = save_sequence(train_set, test_set, last_n_days,
                                            train_seq_path, test_seq_path, last_n_seq_path, train_plain_seq_path, dataSetName)

    # item2vec 处理
    save_item2vec(trainKV, window_size, i2v_path, dataSetName)


if __name__ == "__main__":
    T1 = time.time()
    # readData("diginetica", "./diginetica/data/row/train-item-views.csv")

    # print("./diginetica/data/row/train-item-views.csv".rindex("/"))
    args = sys.argv
    preprocess(args[1], args[2], int(args[3]), int(args[4]), int(args[5]), int(args[6]), int(args[7]))
    # preprocess("diginetica","./diginetica/data/raw/raw.csv",0,15,15,3,7)

    T2 = time.time()
    print("处理时间：{:.0f}:{:.0f}".format((T2-T1)//60, (T2-T1)%60))


















