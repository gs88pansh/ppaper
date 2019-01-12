import os
import numpy as np
import shutil
import time
import tensorflow as tf
from model_ri2v import *
from model_reri2v import *

"""读取 会话 数据"""
def readSessionData(path,sample):
    item_arr_arr = []
    j = 0
    with open(path, "r") as f:
        for fLine in f:
            j += 1
            if (sample and j >= 50000):
                break
            strarr = fLine.split(",")
            if len(strarr) > 2 :
                item_arr = [int(v) for v in strarr[1:]]
                item_arr_arr.append(item_arr)
    return item_arr_arr

def judgeFileExistAndDelCreateSlash(path):
    judgeFileExistAndDelCreate(path, '/')

def judgeFileExistAndDelCreateConvertSlash(path):
    judgeFileExistAndDelCreate(path, '\\')

"""判断文件是否存在，如果存在，就删除，不存在则建立新的文件"""

def judgeFileExistAndDelCreate(path, ch):
    dir = path[0: path.rindex(ch) + 1]
    isExists = os.path.exists(dir)
    if (isExists) :
        if (os.path.isfile(path)) :
            os.remove(path)
    else:
        os.mkdir(dir)
    file = open(path, 'w')
    file.close()

"""判断目录是否存在，如果存在，就删除，不存在则建立新的文件"""
def judgeDirExistAndDelCreate(path):
    isExists = os.path.exists(path)
    if (not isExists):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
    isExists = os.path.exists(path)
    if (not isExists):
        os.makedirs(path)

def judgeDirExistAndDel(path):
    isExists = os.path.exists(path)
    if (isExists):
        shutil.rmtree(path)

"""读取回话的session,固定的读取函数"""
def readSessionDataPost(path):
    with open(path) as file_reader:
        items = []
        itemsl = []
        for line in file_reader.readlines():
            strs = line.split("\n")[0].split(",")
            strs = strs[1:]
            index_arr = [int(v) for v in strs]
            b = index_arr
            if (len(b) >= 2):
                items.extend(b)
                itemsl.append(b)
    print("items 前5个： ", items[:5])
    return itemsl, items

"""将session按照session长度进行分类"""
def classfiedBySessionLength(item_arr_arr):
    dict_ = {}
    items = []
    for arr in item_arr_arr:
        session_length = len(arr)
        if session_length in dict_:
            dict_[session_length].append(arr)
        else:
            dict_[session_length] = [arr]
    for key in dict_:
        print("session长度为:{} ".format(key), "session个数为: {}".format(len(dict_[key])))
    return dict_

"""从已经分类的回话生成输入数据和标签数据以及回话的最后一个标签数据"""
def getInputsAndLabels(dict_):
    train_x = []
    train_y = []
    train_lasty = []

    for key in dict_:
        session_list = dict_[key]
        np_array = np.array(session_list)
        x = [w[:-1] for w in np_array]
        y = [w[1:] for w in np_array]
        lasty = [w[-1] for w in np_array]

        x = np.array(x)
        y = np.array(y)
        lasty = np.array(lasty)
        train_x.append(x)
        train_y.append(y)
        train_lasty.append(lasty)

    return train_x, train_y, train_lasty

# 计算数组中 后面n个数的平均值
def last_n_ave(array,n):
    m = 0
    if (n > len(array)) :
        n = len(array)
    for i in range(n):
        j = -1 - i
        m += array[j]
    return float(m)/n

def log(path, fline):
    with open(path, "a+") as file:
        file.write(fline + "\n")

def trainRnnProcess(args, model_name):
    with tf.Session() as sess:
        judgeDirExistAndDel(args.saver_dir)
        judgeDirExistAndDel(args.summary_path)
        model = None
        if model_name == "ri2v":
            model = RI2V(sess, args)
        elif model_name == "reri2v":
            model = Re_RI2V(sess, args)
        else:
            exit(0)
        D = args.D
        tr_x, tr_y, tr_lasty, tr_items, tr_item_arr_arr = D.input_data(path=args.training_seq_path)
        te_x, te_y, te_lasty, te_items, te_item_arr_arr = D.input_data(path=args.testing_seq_path)

        testX,testY,te_lastY = D.epoch_shuffle_data(te_x,te_y,te_lasty,model.batch_size)
        log(args.log_dir,
            "-----{}_train embedding_size:{} hidden_units:{} batch_size:{} n_sampled:{} drop_out:{:.2f} learning_rate:{:.6f}"
            .format(model_name, args.embedding_size, args.hidden_size, args.batch_size, args.n_sampled, 1-args.keep_prob, args.learning_rate))

        training_num = 0
        for e in range(1,args.epochs + 1):
            costs = []
            T1 = time.time()
            X, Y, lastY = D.epoch_shuffle_data(tr_x, tr_y,tr_lasty, model.batch_size)
            # print(len(X))
            rand_arr = np.arange(len(X))
            np.random.shuffle(rand_arr)

            # 训练过程
            for j in range(len(rand_arr)):
                i = rand_arr[j]
                i_X = X[i]
                i_Y = Y[i]
                i_lastY = lastY[i]

                feed = {model.inputs: i_X, model.labels: i_Y,model.last_labels: i_lastY, model.keep_prob: args.keep_prob}
                _, cost_ = sess.run([model.train_op, model.loss_neg_sample], feed_dict=feed)
                training_num += 1
                costs.append(cost_)

                if j % 5000 == 0:
                    feed = {model.inputs: i_X, model.labels: i_Y,model.last_labels: i_lastY, model.keep_prob: 1}
                    tr_last_1,tr_last_5,tr_last_20,tr_last_total,tr_top_1,tr_top_5,tr_top_20,tr_total = sess.run(
                        [model.last_1, model.last_5,model.last_20,model.last_total,model.top_1,model.top_5,model.top_20,model.total]
                        , feed_dict=feed)

                    print( time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                            j,
                        "{:.4f}".format(tr_last_1/float(tr_last_total)),
                        "{:.4f}".format(tr_last_5/float(tr_last_total)),
                        "{:.4f}".format(tr_last_20/float(tr_last_total)),
                        "{:.4f}".format(tr_top_1/float(tr_total)),
                        "{:.4f}".format(tr_top_5/float(tr_total)),
                        "{:.4f}".format(tr_top_20/float(tr_total))
                    )
                if training_num % 1000 == 0:
                    feed = {model.inputs: i_X, model.labels: i_Y,model.last_labels: i_lastY, model.keep_prob: 1}
                    model.log(feed, training_num)

            # 测试过程
            if e % 5 == 0:
                # epoch_test
                top_1, top_5, top_10, top_20, total, \
                last_1, last_5, last_10, last_20,last_total,\
                test_cost,\
                mrr_at_20, mrr_at_10, mrr_at_5,\
                last_mrr_at_20, last_mrr_at_10, last_mrr_at_5, \
                mrrs = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                for r in range(len(testX)):
                    #r_index = random.randint(r * len_div_tb, (r + 1) * len_div_tb)
                    r_index = r
                    t_X = testX[r_index]
                    t_Y = testY[r_index]
                    t_lastY = te_lastY[r_index]

                    feed = {model.inputs: t_X, model.labels: t_Y,model.last_labels:t_lastY, model.keep_prob:1}

                    tr_last_1, tr_last_5, tr_last_10, tr_last_20, tr_last_total, \
                    tr_top_1, tr_top_5,tr_top_10, tr_top_20, tr_total, \
                    cost_, \
                    mrr_at_5_, mrr_at_10_, mrr_at_20_, \
                    last_mrr_at_5_, last_mrr_at_10_, last_mrr_at_20_ = \
                        sess.run(
                        [model.last_1, model.last_5, model.last_10, model.last_20, model.last_total,
                         model.top_1, model.top_5, model.top_10, model.top_20, model.total,
                         model.loss_neg_sample,
                         model.last_mrr_at_5, model.last_mrr_at_10, model.last_mrr_at_20,
                         model.mrr_at_5, model.mrr_at_10, model.mrr_at_20],
                        feed_dict=feed)

                    top_1 = top_1 + tr_top_1
                    top_5 = top_5 + tr_top_5
                    top_10 = top_10 + tr_top_10
                    top_20 = top_20 + tr_top_20
                    total = total + tr_total

                    last_1 = last_1 + tr_last_1
                    last_5 = last_5 + tr_last_5
                    last_10 = last_10 + tr_last_10
                    last_20 = last_20 + tr_last_20
                    last_total = last_total + tr_last_total

                    mrrs = mrrs + 1
                    mrr_at_20 = mrr_at_20 + mrr_at_20_
                    mrr_at_10 = mrr_at_10 + mrr_at_10_
                    mrr_at_5 = mrr_at_5 + mrr_at_5_
                    last_mrr_at_20 = last_mrr_at_20 + last_mrr_at_20_
                    last_mrr_at_10 = last_mrr_at_10 + last_mrr_at_10_
                    last_mrr_at_5 = last_mrr_at_5 + last_mrr_at_5_

                    test_cost = test_cost + cost_ * tr_total

                t1 = top_1 / float(total)
                t5 = top_5 / float(total)
                t10 = top_10 / float(total)
                t20 = top_20 / float(total)

                l1 = last_1 / float(last_total)
                l5 = last_5 / float(last_total)
                l10 = last_10 / float(last_total)
                l20 = last_20 / float(last_total)
                tloss = test_cost / float(total)

                mrr_at_20 = mrr_at_20 / mrrs
                mrr_at_10 = mrr_at_10 / mrrs
                mrr_at_5 = mrr_at_5 / mrrs


                last_mrr_at_20 = last_mrr_at_20 / mrrs
                last_mrr_at_10 = last_mrr_at_10 / mrrs
                last_mrr_at_5 = last_mrr_at_5 / mrrs


                T2 = time.time()

                print("time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                      "duration: {:.1f}".format(T2 - T1),
                      "epoch: {}".format(e),
                      "{:.3f}".format(t1),
                      "{:.3f}".format(t5),
                      "{:.3f}".format(t10),
                      "{:.3f}".format(t20),
                      "{:.3f}".format(l1),
                      "{:.3f}".format(l5),
                      "{:.3f}".format(l10),
                      "{:.3f}".format(l20),
                      "mrr5:{:.3f}".format(mrr_at_5),
                      "mrr10:{:.3f}".format(mrr_at_10),
                      "mrr20:{:.3f}".format(mrr_at_20),
                      "lmrr5:{:.3f}".format(last_mrr_at_5),
                      "lmrr10:{:.3f}".format(last_mrr_at_10),
                      "lmrr20:{:.3f}".format(last_mrr_at_20),
                      "cost:", "{:.4f}".format(last_n_ave(costs, 20)),
                      "test_cost:", "{:.4f}".format(tloss)
                      )

                log(args.log_dir, "{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
                    e,
                    t1, t5, t10, t20,
                    l1, l5, l10, l20,
                    mrr_at_5, mrr_at_10, mrr_at_20,
                    last_mrr_at_5, last_mrr_at_10, last_mrr_at_20,
                    last_n_ave(costs, 20), tloss))
                model.save(e)

def testRnnProcess(args, model_name):
    with tf.Session() as sess:
        model = Re_RI2V(args, sess)

        D = args.D
        te_x, te_y, te_lasty, te_items, te_item_arr_arr = D.input_data(path=args.training_seq_path)
        testX,testY,te_lastY = D.epoch_shuffle_data(te_x,te_y,te_lasty,model.batch_size)

        log(args.log_dir,
            "-----{}_test embedding_size:{} hidden_units:{} batch_size:{} n_sampled:{} drop_out:{:.2f} learning_rate:{:.6f}"
            .format(model_name, args.embedding_size, args.hidden_size, args.batch_size, args.n_sampled, 1-args.keep_prob, args.learning_rate))

        top_1, top_5, top_10, top_20, total, \
        last_1, last_5, last_10, last_20,last_total,\
        test_cost,\
        mrr_at_20, mrr_at_10, mrr_at_5,\
        last_mrr_at_20, last_mrr_at_10, last_mrr_at_5, \
        mrrs = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        for r in range(len(testX)):
            #r_index = random.randint(r * len_div_tb, (r + 1) * len_div_tb)
            r_index = r
            t_X = testX[r_index]
            t_Y = testY[r_index]
            t_lastY = te_lastY[r_index]

            feed = {model.inputs: t_X, model.labels: t_Y,model.last_labels:t_lastY, model.keep_prob:1}

            tr_last_1, tr_last_5, tr_last_10, tr_last_20, tr_last_total, \
            tr_top_1, tr_top_5,tr_top_10, tr_top_20, tr_total, \
            cost_, \
            mrr_at_5_, mrr_at_10_, mrr_at_20_, \
            last_mrr_at_5_, last_mrr_at_10_, last_mrr_at_20_ = \
                sess.run(
                [model.last_1, model.last_5, model.last_10, model.last_20, model.last_total,
                 model.top_1, model.top_5, model.top_10, model.top_20, model.total,
                 model.loss_neg_sample,
                 model.last_mrr_at_5, model.last_mrr_at_10, model.last_mrr_at_20,
                 model.mrr_at_5, model.mrr_at_10, model.mrr_at_20],
                feed_dict=feed)

            top_1 = top_1 + tr_top_1
            top_5 = top_5 + tr_top_5
            top_10 = top_10 + tr_top_10
            top_20 = top_20 + tr_top_20
            total = total + tr_total

            last_1 = last_1 + tr_last_1
            last_5 = last_5 + tr_last_5
            last_10 = last_10 + tr_last_10
            last_20 = last_20 + tr_last_20
            last_total = last_total + tr_last_total

            mrrs = mrrs + 1
            mrr_at_20 = mrr_at_20 + mrr_at_20_
            mrr_at_10 = mrr_at_10 + mrr_at_10_
            mrr_at_5 = mrr_at_5 + mrr_at_5_
            last_mrr_at_20 = last_mrr_at_20 + last_mrr_at_20_
            last_mrr_at_10 = last_mrr_at_10 + last_mrr_at_10_
            last_mrr_at_5 = last_mrr_at_5 + last_mrr_at_5_

            test_cost = test_cost + cost_ * tr_total

        t1 = top_1 / float(total)
        t5 = top_5 / float(total)
        t10 = top_10 / float(total)
        t20 = top_20 / float(total)

        l1 = last_1 / float(last_total)
        l5 = last_5 / float(last_total)
        l10 = last_10 / float(last_total)
        l20 = last_20 / float(last_total)
        tloss = test_cost / float(total)

        mrr_at_20 = mrr_at_20 / mrrs
        mrr_at_10 = mrr_at_10 / mrrs
        mrr_at_5 = mrr_at_5 / mrrs


        last_mrr_at_20 = last_mrr_at_20 / mrrs
        last_mrr_at_10 = last_mrr_at_10 / mrrs
        last_mrr_at_5 = last_mrr_at_5 / mrrs


        T2 = time.time()

        print("time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
              "{:.3f}".format(t1),
              "{:.3f}".format(t5),
              "{:.3f}".format(t10),
              "{:.3f}".format(t20),
              "{:.3f}".format(l1),
              "{:.3f}".format(l5),
              "{:.3f}".format(l10),
              "{:.3f}".format(l20),
              "mrr5:{:.3f}".format(mrr_at_5),
              "mrr10:{:.3f}".format(mrr_at_10),
              "mrr20:{:.3f}".format(mrr_at_20),
              "lmrr5:{:.3f}".format(last_mrr_at_5),
              "lmrr10:{:.3f}".format(last_mrr_at_10),
              "lmrr20:{:.3f}".format(last_mrr_at_20),
              "cost:", "{:.4f}".format(0),
              "test_cost:", "{:.4f}".format(tloss)
              )

        log(args.log_dir,
            "{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}"
            .format(
                "test",
                t1, t5, t10, t20,
                l1, l5, l10, l20,
                mrr_at_5, mrr_at_10, mrr_at_20,
                last_mrr_at_5, last_mrr_at_10, last_mrr_at_20, 0, tloss
                )
            )


def evaluate_sessions(pr, metrics, test_data, train_data, items=None, cut_off=20, session_key='SessionId',
                      item_key='ItemId', time_key='Time'):
    '''
    Evaluates the algorithms wrt. the given metrics. Has no batch evaluation capabilities.

    Parameters
    --------
    pr : 训练好的实例
    metrics : list
    test_data : pandas.DataFrame
        session IDs, item IDs and timestamp (unix timestamps).
    train_data : pandas.DataFrame
    items : 1D list or None
        你想要比较分数的物品列表. 如果是None，所有的训练集都用上
    cut-off : int
        推荐列表长度，N recall@N
    session_key : string
        (default: 'SessionId')
    item_key : string
        (default: 'ItemId')
    time_key : string
        (default: 'Time')

    Returns
    --------
    out :  list of tuples
        (metric_name, value)

    '''

    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    count = 0
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')

    sc = time.clock()
    st = time.time()

    time_sum = 0
    time_sum_clock = 0
    time_count = 0

    for m in metrics:
        m.reset()

    test_data.sort_values([session_key, time_key], inplace=True)
    items_to_predict = train_data[item_key].unique()
    prev_iid, prev_sid = -1, -1
    for i in range(len(test_data)):

        if count % 1000 == 0:
            print('    eval process: ', count, ' of ', actions, ' actions: ', (count / actions * 100.0), ' % in',
                  (time.time() - st), 's')

        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        if prev_sid != sid:
            prev_sid = sid
        else:
            if items is not None:
                if np.in1d(iid, items):
                    items_to_predict = items
                else:
                    items_to_predict = np.hstack(([iid], items))

            crs = time.clock()
            trs = time.time()
            preds = pr.predict_next(sid, prev_iid, items_to_predict)

            preds[np.isnan(preds)] = 0
            #             preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
            preds.sort_values(ascending=False, inplace=True)

            time_sum_clock += time.clock() - crs
            time_sum += time.time() - trs
            time_count += 1

            for m in metrics:
                m.add(preds, iid)

        prev_iid = iid

        count += 1

    print('END evaluation in ', (time.clock() - sc), 'c / ', (time.time() - st), 's')
    print('    avg rt ', (time_sum / time_count), 's / ', (time_sum_clock / time_count), 'c')
    print('    time count ', (time_count), 'count/', (time_sum), ' sum')

    res = []
    for m in metrics:
        res.append(m.result())
    return res

def dateNow():
    return time.strftime("%Y-%m-%e %H:%M:%S")

if __name__ == "__main__":
    print(os.path.join(os.getcwd(), 'trained_variables2.ckpt'))