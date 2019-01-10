import sys
from pylab import *
sys.path.append("../")

paths = ["../diginetica/result.log"]


def getResults(paths):
    results = []
    for path in paths:
        RECALL_20s = []
        MRR_20s = []
        Xs = []
        descs = []
        with open(path, "r") as file:
            is_data_line = False
            x = []
            m = []
            r = []
            desc = ""
            for fLine in file.readlines():
                if fLine == "" or fLine is None:
                    continue
                if not is_data_line :
                    if fLine.find("-----ri2v_train") == 0:
                        is_data_line = True
                        desc = fLine
                        continue
                if is_data_line:
                    if fLine.find("-----") == 0 or len(fLine) == 0:
                        if len(x) > 0:
                            RECALL_20s.append(r)
                            MRR_20s.append(m)
                            Xs.append(x)
                            descs.append(desc)
                            r, m, x, desc = [],[],[],""
                        if fLine.find("-----ri2v_train") != 0:
                            is_data_line = False
                        else:
                            is_data_line = True
                            desc = fLine
                        continue
                    # 是真实的数据
                    fLines = fLine.split(" ")
                    if int(fLines[0]) % 5 == 0:
                        x.append(int(fLines[0]))
                        m.append(float(fLines[11]))
                        r.append(float(fLines[4]))
            if len(x) > 0:
                RECALL_20s.append(r)
                MRR_20s.append(m)
                Xs.append(x)
                descs.append(desc)
                r, m, x, desc = [],[],[],""
        results.append([Xs, RECALL_20s, MRR_20s, descs])
    return results

def pain(x, y, desc):
    l1, = plt.plot(x, y, "-o")
    return l1

def painResult(results):
    for result in results:
        Xs = result[0]
        Rs = result[1]
        Ms = result[2]
        Ds = result[3]

        plt.figure(1)
        plt.subplot(111)

        L = []
        D = []
        plt.title("Diginetica Recall@20")
        for i in range(len(Xs)):
            x = Xs[i]
            r = Rs[i]
            m = Ms[i]
            d = Ds[i]
            L.append(pain(x, r, d))
            D.append(d)
        plt.legend(handles = L, labels = D, loc = 4)
        plt.yticks(np.linspace(0.50, 0.54, 5))

        plt.figure(2)
        plt.subplot(111)
        L = []
        D = []
        plt.title("Diginetica MRR@20")
        for i in range(len(Xs)):
            x = Xs[i]
            r = Rs[i]
            m = Ms[i]
            d = Ds[i]
            L.append(pain(x, m, d))
            D.append(d)
        plt.legend(handles = L, labels = D, loc = 4)
        plt.yticks(np.linspace(0.18, 0.22, 5))
        plt.show()

if __name__ == "__main__":
    painResult(getResults(paths))






