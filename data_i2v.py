import numpy as np
import time

# 用于数据的读取、每个 epoch 迭代的数据的重新整理
class Item2VecDataSet(object):
    def __init__(self,path1):
        self.path1 = path1

    # 获得item2vec训练数据
    def readDataFromPath(self,path):
        x, y = [],[]
        with open(path, "r") as f:
            for fLine in f:
                strarr = fLine.split("\n")[0].split(",")
                x.append(int(strarr[0]))
                y.append(int(strarr[1]))

        x = np.array(x)
        y = np.array(y)
        return x, y

    def shuffle(self,x,y):
        # shuffle x,y
        all_index = np.arange(len(x))
        np.random.shuffle(all_index)
        x = x[all_index]
        y = y[all_index]
        return x,y

    def input_data(self):
        t1 = time.time()
        x,y = self.readDataFromPath(self.path1)
        t2 = time.time()
        print("time {:.1f} sec".format(t2-t1))
        return x,y

