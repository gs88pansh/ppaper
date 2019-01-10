import sys
sys.path.append("../")
from pylab import *

print("nima")

def timeEffect(x,a,b):
    ea_bx = np.exp(a - x*b)
    ea = np.exp(a)
    return (ea_bx/(ea_bx+1)) *(ea+1)/(ea)

x = np.arange(-50, 100, 0.2)

a_b = [-0.118652,0.0747636]


a_b_change_y = timeEffect(x,a_b[0],a_b[1])


plt.figure(1)
l1, = plt.plot(x,a_b_change_y)
plt.legend(handles=[l1,], labels=['{:.4f} {:.4f}'.format(a_b[0], a_b[1]),
        ], loc=1)
plt.show()


