import numpy as np

# 修改 np 数组中的值
def test_changeNpArrayValue() :
    a = np.zeros([10])
    a[[1,2,3,4,5]] = 2
    print(a)

    a = np.zeros([5, 5])
    print(a)

if __name__ == "__main__":
    test_changeNpArrayValue()
