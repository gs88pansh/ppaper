import numpy as np
from Utils import *



# RI2V 的数据输入模块
class RI2VDataSet(object):
    def epoch_shuffle_data(self,train_x, train_y,train_lasty, batch_size):
        """
        for i in range(len(X)):
            X[i] batch_size * steps
            Y[i] batch_size * steps
        """
        X = []
        Y = []
        LastY = []
        for i in range(len(train_x)):
            x = train_x[i]
            y = train_y[i]
            lasty = train_lasty[i]

            x, y,lasty = self.shuffle3(x, y,lasty)

            x_length = len(x)
            for m in range(int(x_length / batch_size)):
                batch_x = x[m * batch_size:(m + 1) * batch_size]
                batch_y = y[m * batch_size:(m + 1) * batch_size]
                batch_lasty = lasty[m * batch_size:(m + 1) * batch_size]
                X.append(batch_x)
                Y.append(batch_y)
                LastY.append(batch_lasty)
        return X, Y, LastY

    def shuffle3(self,x,y,z):
        all_index = np.arange(len(x))
        np.random.shuffle(all_index)
        x = x[all_index]
        y = y[all_index]
        z = z[all_index]
        return x, y, z

    def shuffle(self,x, y):
        # shuffle x,y
        #     t1 = time.time()
        all_index = np.arange(len(x))
        np.random.shuffle(all_index)
        x = x[all_index]
        y = y[all_index]
        #     t2 = time.time()
        #     print("shuffle耗时：",t2-t1,"sec")
        #     print(x[:20])
        return x, y

    def input_data(self,path):
        item_arr_arr, items = readSessionDataPost(path=path)
        session_length_list = classfiedBySessionLength(item_arr_arr)
        x, y,lasty = getInputsAndLabels(session_length_list)
        return x, y,lasty, items, item_arr_arr
