import sys
import numpy as np
import matplotlib.pyplot as plt

# 曲线图 点击次数商品分布
# x轴 点击次数
# y轴 商品数
#

def readClickCountStatis(path):
    x = []
    y = []
    with open(path, "r") as file:
        for fLine in file:
            arr = fLine.split(',')
            clickCount = int(arr[1])
            clickCountItemCount = int(arr[2])
            x.append(clickCount)
            y.append(clickCountItemCount)
    return x,y

def simplepaint(x,y):
    plt.figure(1)
    plt.subplot(111)
    l1, = plt.plot(x, y)
    plt.legend(handles=[l1], labels=['item count'],loc=1)
    plt.show()

def smooth(x,y,n):
    index = 0
    xarr = [0]
    yarr = [0]
    for i in range(len(x)):
        if i // n == index:
            yarr[i // n] += y[i]
        else:
            index = i // n
            yarr.append(0)
    return range(len(yarr)),yarr



if __name__ == "__main__":

    dataSets = ["diginetica"]
    for data_set_name in dataSets:
        readPath = "../{}/data/analysis/_02numOfClicks.txt".format(data_set_name)
        x,y = readClickCountStatis(readPath)
        #x,y = smooth(x,y,50)
        for i in range(len(x)-1):
            y[i] = y[i+1]
        simplepaint(x,y)
