

"""

time

EffectTimeLine

ClickTimeLine


"""
from pylab import *
def timeEffect(x,a,b):
    ea_bx = np.exp(a - x*b)
    ea = np.exp(a)
    return (ea+1)/(ea)*(ea_bx/(ea_bx+1))

a1, b1, a2, b2 = -0.0726218,0.108536 , -2.45967e-05,0.0153205
clickTimeLine1 = [8, 8, 6, 12, 6, 7, 5, 3, 12, 7, 12, 7, 8, 7, 7, 3, 1, 7, 5, 3, 6, 6, 5, 4, 5, 8, 12, 7, 7, 10, 13, 5, 8, 11, 1, 1, 8, 7, 7, 3, 8, 4, 9, 5, 10, 5, 6, 3, 4, 4, 3, 3, 7, 6, 3, 2, 1, 1, 4, 2, 5, 29, 327, 16, 73, 54, 28, 25, 36, 32, 38, 30, 19, 29, 57, 195, 137, 28, 134, 149, 176, 187, 377, 60, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 1, 4, 1, 1, 3, 6, 6, 4, 1, 1, 4, 1, 1, 1, 1, 3, 1, 9, 17, 18, 14]
clickTimeLine2 = [43, 4, 24, 21, 18, 17, 36, 21, 4, 21, 19, 19, 23, 21, 11, 3, 27, 30, 17, 20, 21, 23, 9, 27, 19, 12, 13, 17, 29, 5, 30, 34, 23, 25, 36, 35, 12, 23, 34, 37, 24, 35, 32, 18, 30, 31, 28, 48, 77, 29, 16, 45, 31, 33, 21, 42, 28, 13, 22, 31, 25, 43, 22, 37, 19, 40, 31, 44, 35, 44, 39, 5, 9, 27, 27, 19, 48, 27, 15, 16, 15, 25, 26, 44, 41, 19, 20, 20, 28, 35, 47, 49, 15, 18, 21, 17, 20, 7, 5]
clickTimeLine1 = clickTimeLine1[::-1]
clickTimeLine2 = clickTimeLine2[::-1]

time1 = np.arange(0, len(clickTimeLine1), 1)
time2 = np.arange(0, len(clickTimeLine2), 1)
timeEy1 = timeEffect(time1,a1,b1)
timeEy2 = timeEffect(time2,a2,b2)

fig = plt.figure(1)
ax1 = fig.add_subplot(211)
l1, = ax1.plot(time1,timeEy1, "r-")
ax1.set_ylabel('$f^*(x)$')
ax2 = ax1.twinx()
l2, = ax2.plot(time1, clickTimeLine1)
ax2.set_ylabel('item clicks')
plt.legend(handles=[l1,l2, ], labels=['$f^*(x)$', '#clicks'], loc=1)

ax1 = fig.add_subplot(212)
l1, = ax1.plot(time2,timeEy2, "r-")
ax1.set_ylabel('$f^*(x)$')
ax2 = ax1.twinx()
l2, = ax2.plot(time2, clickTimeLine2)
ax2.set_ylabel('item clicks')
plt.legend(handles=[l1,l2, ], labels=['$f^*(x)$', '#clicks'], loc=1)

plt.show()






