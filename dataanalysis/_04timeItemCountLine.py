# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt


def readFile(path):
    map = {}
    arrarr = []
    with open(path, "r") as file:
        for fLine in file:
            strarr = fLine.split(',')
            itemId = strarr[1].split('_')[0]
            itemType = strarr[1].split('_')[1]
            if (len(strarr) < 1000):
                continue

            times = []
            timeF = float(strarr[2])
            for i in  range(len(strarr)-2):
                k = i+2
                time = float(strarr[k]) - timeF
                times.append(time)
            arrarr.append(times)
    return arrarr


def getLine(arr):
    rarr = [0]
    index = 0
    for i in range(len(arr)):
        if (arr[i] // (24 * 60 * 60) == index):
            rarr[index] += 1
        else:
            rarr.append(1)
            index += 1
    print("共有："+str(index))
    return rarr,index

def simplepaint(x,y):
    plt.figure(1)
    plt.subplot(111)
    l1, = plt.plot(x, y, ":bo")
    plt.legend(handles=[l1], labels=['#item'],loc=1)
    plt.show()

if __name__ == "__main__":
    readpath = "D:\\桂越\\我的论文\\实验数据集\\preprocess\\物品点击时间映射.txt"
    arrarr = readFile(readpath)
    step = len(arrarr) // 11
    i = 0
    plt.figure(1)
    k = 1
    while(i < len(arrarr) and k <=9):
        arr = arrarr[i]
        line,days = getLine(arr)
        if days < 60:
            i=i+1
            continue
        plt.subplot(int("33" + str(k)))
        if k == 4 or k == 7:
            print(line,":" ,days)
        k += 1
        l1, = plt.plot(range(len(line)), line)
        plt.legend(handles=[l1], labels=['#item'],loc=1)

        i = i + step
    plt.show()
