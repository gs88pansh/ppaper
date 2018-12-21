import sys
sys.path.append("../")
from pylab import *

print("nima")

def timeEffect(x,a,b):
    ea_bx = np.exp(a - x*b)
    ea = np.exp(a)
    return (ea+1)/(ea)*(ea_bx/(ea_bx+1))

x = np.arange(-10, 10, 1)

a_b = [
[0.0, 0.074413389] ,
[-0.058111977, 0.20559169] ,
[-0.17408749, 0.34014213] ,
[-0.3038491, 0.47300375] ,
[-0.43998623, 0.60040468] ,
[-0.57560331, 0.71855682] ,
[-0.70289278, 0.82396954] ,
[-0.81435555, 0.91404176] ,
[-0.90456116, 0.98741466] ,
[-0.97097224, 1.0439335] ,
]


a_b_change_y = [timeEffect(x,ab[0],ab[1]) for ab in a_b]


plt.figure(1)
plt.subplot(211)
l1, = plt.plot(x,a_b_change_y[0])
l2, = plt.plot(x,a_b_change_y[1])
l3, = plt.plot(x,a_b_change_y[2])
l4, = plt.plot(x,a_b_change_y[3])
plt.legend(handles=[l1,l2,l3,l4], labels=['{:.2f} {:.2f}'.format(a_b[0][0], a_b[0][1]),
            '{:.2f} {:.2f}'.format(a_b[1][0], a_b[1][1]),
            '{:.2f} {:.2f}'.format(a_b[2][0], a_b[2][1]),
            '{:.2f} {:.2f}'.format(a_b[3][0], a_b[3][1]),
        ], loc=1)

plt.subplot(212)
l1, = plt.plot(x,a_b_change_y[4])
l2, = plt.plot(x,a_b_change_y[5])
l3, = plt.plot(x,a_b_change_y[6])
l4, = plt.plot(x,a_b_change_y[7])
plt.legend(handles=[l1,l2,l3,l4], labels=['{:.2f} {:.2f}'.format(a_b[4][0], a_b[4][1]),
            '{:.2f} {:.2f}'.format(a_b[5][0], a_b[5][1]),
            '{:.2f} {:.2f}'.format(a_b[6][0], a_b[6][1]),
            '{:.2f} {:.2f}'.format(a_b[7][0], a_b[7][1]),
        ], loc=1)
plt.show()


