import time
from Utils import *

## 生成 item -> time
## 生成物品 -> 时间 的文件，格式：index,物品id_类别,timestamp1,timestamp2,...





def getTime(timestr, strpattern="%Y-%m-%d %H:%M:%S"):
    timestamp = time.mktime(time.strptime(timestr, strpattern))
    #print(timestamp)
    return timestamp

def readItemClickTimeYoochoose(path, map):
    with open(path,"r") as f:
        i = 0
        for fLine in f:
            i= i + 1
            if i % 500000 == 0 :
                print(i)
            #print(fLine)
            strarr = fLine.split(',')
            strTime = strarr[1]
            strItemId = strarr[2]+"_"+strarr[3].split('\n')[0]
            if (strItemId in map) :
                map.get(strItemId).append(getTime(strTime, "%Y-%m-%d"))
            else:
                map[strItemId] = [getTime(strTime, "%Y-%m-%d")]

def readItemClickTimeDiginetica(path, map):
    with open(path,"r") as f:
        i = 0
        for fLine in f:
            i= i + 1
            if i == 1:
                continue
            if i % 500000 == 0 :
                print(i)
            #print(fLine)
            strarr = fLine.split(';')
            strTime = strarr[4].split("\n")[0]
            strItemId = strarr[2]
            if (strItemId in map) :
                map.get(strItemId).append(getTime(strTime, "%Y-%m-%d"))
            else:
                map[strItemId] = [getTime(strTime, "%Y-%m-%d")]

def readItemClickTime(path, map, datasetname):
    if datasetname == "yoochoose":
        return readItemClickTimeYoochoose(path, map)
    if datasetname == "diginetica":
        return readItemClickTimeDiginetica(path, map)
    return None

def writeItemTimeList(path, map):
    judgeFileExistAndDelCreateSlash(path)
    with open(path, "a+") as f:
        i = 0
        strr = ""
        for k in map.keys():
            i = i + 1
            strr += str(i) + "," + k
            map.get(k).sort()
            for stamp in map.get(k):
                strr += "," + str(stamp)
            strr += "\n"
            if (i % 1000 == 0):
                f.write(strr)
                strr = ""
        f.write(strr)

if __name__ == '__main__':

    # yoochoose
    # map = {}
    # readPath = "../yoochoose/data/raw/yoochoose-clicks.dat"
    # readItemClickTime(readPath, map, "yoochoose")
    # writePath = "../yoochoose/data/analysis/_01itemClickTimeMap.txt"
    # writeItemTimeList(writePath, map)

    # diginetica
    map = {}
    readPath = "../diginetica/data/raw/train-item-views.csv"
    readItemClickTime(readPath, map, "diginetica")
    writePath = "../diginetica/data/analysis/_01itemClickTimeMap.txt"
    writeItemTimeList(writePath, map)




