import sys
sys.path.append("./")
from Utils import *
from model_i2v import *
from data_i2v import *

args = sys.argv
# item_size = 43136  # item_size总的个数
# batch_size = 512  # batch大小
# embedding_size = 100  # embedding_size 30 50 100
# n_sampled = 256  # 采样个数
# epochs = 40  # 迭代轮数
# path1 = WINDOW_3_TRAINING_FILE  # item2vec训练集文件路径
# learning_rate = 0.0007

print("后面的依次为参数："
      "\n \t n_sampled", args[1],
      "\n \t learning_rate", args[2],
      "\n \t epochs", args[3],
      "\n \t embedding_size", args[4],
      "\n \t batch_size", args[5],
      "\n \t item_size", args[6],
      "\n \t train_file", args[7],
      "\n \t model_base_dir", args[8])

if __name__ == "__main__":

    n_sampled = int(args[1])  # 采样个数
    learning_rate = float(args[2])
    epochs = int(args[3])  # 迭代轮数
    embedding_size = int(args[4])  # embedding_size 30 50 100
    batch_size = int(args[5])  # batch大小
    item_size = int(args[6])  # item_size总的个数
    path1 = args[7]  # item2vec训练集文件路径
    log_dir = args[8] + "/README.txt"

    # model save path
    modelPath = args[8] + "/i2v/" \
                "/em{}ba{}sa{}le{}".format(embedding_size, batch_size, n_sampled, learning_rate)
    # 损失函数存储文件
    viewLossPath = args[8] + "/i2v/"\
                   "/view_em{}ba{}sa{}le{}".format(embedding_size, batch_size, n_sampled, learning_rate)

    judgeDirExistAndDel(modelPath)
    judgeDirExistAndDel(viewLossPath)

    # 生成模型
    model = item2vec(embedding_size,n_sampled,item_size,learning_rate=learning_rate)

    # 获得数据
    D = Item2VecDataSet(path1)
    x,y = D.input_data()

    print("training data length: {}" .format(len(x)))
    print("batches: {}".format(len(x) // batch_size))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merge_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(viewLossPath, tf.get_default_graph())
        train_costs = []


        log(log_dir,
                    "-----item2vec embedding_size:{} batch_size:{} n_sampled:{} learning_rate:{}"
            .format(embedding_size, batch_size, n_sampled, learning_rate))
        for e in range(1, epochs + 1):
            x, y = D.shuffle(x, y)
            start = int(time.time())
            idx_start = 0
            idx_end = batch_size
            for i in range(int(len(x) / batch_size)):
                # print(i)

                batch_x = x[idx_start:idx_end]
                batch_y = y[idx_start:idx_end]
                idx_start = idx_start + batch_size
                idx_end = idx_end + batch_size
                feed = {
                    model.inputs: batch_x,
                    model.labels: batch_y[:, None],
                    model.keep_prob : 1
                }
                cost_,_ = sess.run([model.cost, model.optimizer], feed_dict=feed)

                if i%500 == 0:
                    train_costs.append(cost_)
                    su = sess.run(merge_op,feed_dict=feed)
                    summary_step = (e - 1) * (int(len(x) / batch_size)) + i
                    summary_writer.add_summary(su,summary_step)

            end = int(time.time())
            train_costs.append(1)
            # plt.figure()
            # l1, = plt.plot(range(len(train_costs)),train_costs)
            # plt.legend(loc='cost')
            # plt.show()

            lastNCost = last_n_ave(train_costs,50)
            print("time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                  "Epoch{}".format(e),
                  "duration: {}:{} ".format((end - start) // 60, (end - start) % 60),
                  "cost: {:.3f}".format(lastNCost))

            log(log_dir, "{} {:.3f}".format(e, lastNCost))

            start = int(time.time())
            model.saver.save(sess, modelPath + "/A{}".format(e))
