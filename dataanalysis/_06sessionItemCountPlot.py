
import matplotlib.pyplot as plt

def xItemCountySessionCount(path):
    maxNum = 0
    map = {}
    with open(path, "r") as f:
        for fLine in f:
            strarr = fLine.split(",")
            countstr = strarr[1]
            if (countstr in map):
                map[countstr] = map.get(countstr) + 1
            else:
                map[countstr] = 1

            count = int(countstr,10)

            if (maxNum < count) :
                maxNum = count
    x = [i for i in range(maxNum + 1)]
    y = [0 for i in range(maxNum + 1)]
    for key in map.keys():
        y[int(key)] = map.get(key)
    return x,y


def simplepaint(x, y):
    plt.figure(1)
    plt.subplot(111)
    l1, = plt.plot(x, y, ":bo")
    plt.legend(handles=[l1], labels=['session count'], loc=1)
    plt.show()

if __name__ == "__main__":

    # diginetica
    path = "../diginetica/data/analysis/_05sessionLength_sessionNum.txt"
    x,y = xItemCountySessionCount(path)
    simplepaint(x,y)



