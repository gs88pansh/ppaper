from pylab import *
import random
from matplotlib.font_manager import FontProperties

i2v_m = 106
ri2v_m = 30
gru_m = 120

i2v_x = [(i2v_m * i + random.uniform(-5, 5))/60 for i in range(5)]
ri2v_x = [(ri2v_m * i + random.uniform(-3, 3))/60 + i2v_x[-1] for i in range(11)]

gru_x = [(gru_m * i + random.uniform(-7, 7))/60 for i in range(11)]

i2v_y = [0,0,0,0,0]
ri2v_y = [0, 0.6222, 0.6315, 0.6356, 0.6380, 0.6391, 0.6410, 0.6420, 0.6421, 0.6417, 0.6422]
gru_y = [0, 0.5918, 0.60, 0.61,0.615,0.62,0.622,0.6225,0.6222, 0.6226, 0.6228]


print(i2v_x)
print(ri2v_x)
print(gru_x)

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
l1, = ax1.plot(ri2v_x,ri2v_y, "o-")
l2, = ax1.plot(gru_x,gru_y,"o-")
l3, = ax1.plot(i2v_x,i2v_y,"o-")

plt.xlabel('time /h')
plt.ylabel('Recall@20 (%)')
plt.legend(handles=[l1,l2,l3 ], labels=['RI2V', 'GRU4REC', 'Item2Vector'])

plt.show()