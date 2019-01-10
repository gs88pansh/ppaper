from Utils import *

# 统计商品的个数
# 统计session的个数
# 统计商品点击次数
# 平均每个商品点击次数
# 平均会话长度

def statisItems(path, dataSetName) :
    if dataSetName == "yoochoose":
        return statisItemsYoochoose(path)
    if dataSetName == "diginetica":
        return statisItemsDiginetica(path)

def statisItemsDiginetica(path):
    sessionMap = {}
    sessionTimeMap = {}
    itemMap = {}
    with open(path,"r") as f:
        i = 0
        for fLine in f:
            i= i + 1
            if i == 1:
                continue
            if (i % 10000000 == 0):
                print(str(i))
            arr = fLine.split(';')
            # 统计相同session
            if (arr[0] in sessionMap):
                sessionMap[arr[0]] = sessionMap.get(arr[0]) + 1
            else:
                sessionMap[arr[0]] = 1

            # 统计相同物品
            if (arr[2] in itemMap):
                itemMap[arr[2]] = itemMap.get(arr[2]) + 1
            else:
                itemMap[arr[2]] = 1

            # 统计session发生的时间
            if (arr[4] in sessionTimeMap):
                continue
            else:
                sessionTimeMap[arr[0]] = arr[1]
        print("Diginetica总点击数：" + str(i))
        print("Diginetica总会话数：" + str(len(sessionMap)))
        print("Diginetica总商品数：" + str(len(itemMap)))
        return itemMap,sessionMap,sessionTimeMap

def statisItemsYoochoose(path):
    sessionMap = {}
    sessionTimeMap = {}
    itemMap = {}
    with open(path,"r") as f:
        i = 0
        for fLine in f:
            i= i + 1
            if (i % 10000000 == 0):
                print(str(i))
            arr = fLine.split(',')
            # 统计相同session
            if (arr[0] in sessionMap):
                sessionMap[arr[0]] = sessionMap.get(arr[0]) + 1
            else:
                sessionMap[arr[0]] = 1

            # 统计相同物品
            if (arr[2] in itemMap):
                itemMap[arr[2]] = itemMap.get(arr[2]) + 1
            else:
                itemMap[arr[2]] = 1

            # 统计session发生的时间
            if (arr[0] in sessionTimeMap):
                continue
            else:
                sessionTimeMap[arr[0]] = arr[1]
        print("Yoochoose总点击数：" + str(i))
        print("Yoochoose总会话数：" + str(len(sessionMap)))
        print("Yoochoose总商品数：" + str(len(itemMap)))
        return itemMap,sessionMap,sessionTimeMap

def writeCount(path, map):
    judgeFileExistAndDelCreateSlash(path)
    with open(path, "a+") as f:
        i = 0
        strr = ""
        for k in map.keys():
            i = i + 1
            strr += k + "," + str(map.get(k)) + "\n"
            if (i % 1000 == 0):
                f.write(strr)
                strr = ""
        f.write(strr)

if __name__ == "__main__" :

    # yoochoose
    # readPath = "../yoochoose/data/rawyoochoose-clicks.dat"
    # itemMap, sessionMap,sessionTimeMap = statisItems(readPath)
    # # writePath = "D:\\桂越\\我的论文\\实验数据集\\preprocess\\itemCount.txt"
    # # writeCount(writePath, itemMap)
    # writePath = "../yoochoose/data/analysis/sessionLength_sessionNum.txt"
    # writeCount(writePath, sessionMap)
    # writePath = "../yoochoose/data/analysis/sessionTime_sessionNum.txt"
    # writeCount(writePath, sessionTimeMap)

    # diginetica
    readPath = "../diginetica/data/raw/train-item-views.csv"
    itemMap, sessionMap,sessionTimeMap = statisItems(readPath, "diginetica")
    # writePath = "D:\\桂越\\我的论文\\实验数据集\\preprocess\\itemCount.txt"
    # writeCount(writePath, itemMap)
    writePath = "../diginetica/data/analysis/_05sessionLength_sessionNum.txt"
    writeCount(writePath, sessionMap)
    # writePath = "../diginetica/data/analysis/sessionTime_sessionNum.txt"
    # writeCount(writePath, sessionTimeMap)




