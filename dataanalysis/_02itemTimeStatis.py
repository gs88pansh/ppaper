
# 相同物品的点击次数统计


def readFile(path):
    map = {}
    with open(path, "r") as file:
        for fLine in file:
            strarr = fLine.split(',')
            if (len(strarr)-2) in map :
                map[len(strarr)-2] = map[len(strarr)-2]+1
            else:
                map[len(strarr) - 2] = 1
    return map


def writeItemStatisList(path, arrarr):
    with open(path, "a+") as f:
        i = 0
        strr = ""
        for k in arrarr:
            i = i + 1
            strr += str(i) + "," + str(k[0]) + "," + str(k[1])
            strr += "\n"
            if (i % 1000 == 0):
                f.write(strr)
                strr = ""
        f.write(strr)


if __name__ == "__main__":
    datasets = ["diginetica"]
    for data_set_name in datasets:
        readpath = "../{}/data/analysis/_01itemClickTimeMap.txt".format(data_set_name)
        map = readFile(readpath)
        writepath = "../{}/data/analysis/_02numOfClicks.txt".format(data_set_name)
        arrarr = []
        for key in map.keys():
            arrarr.append([key,map[key]])
        arrarr.sort(key=lambda x:x[0])
        writeItemStatisList(writepath, arrarr)





