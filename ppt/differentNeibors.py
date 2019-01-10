# -*- coding: utf-8 -*-
"""

SCIV-KNN 不同近邻性能对比, 不适宜在一起的直方图

"""

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


fig = plt.figure(1)

ax1 = fig.add_subplot(221)
ax1.bar([1], [61.28-60], label='SCIV-kNN k=100', bottom=60)
ax1.bar([2], [61.63-60], label='SCIV-kNN k=500', bottom=60)
plt.xticks([0, 1.5, 3], ["", "yoochoose recall@20", ""], rotation=0)
plt.yticks(np.arange(60, 64, 1), rotation=0)
plt.legend(loc=1)
plt.ylabel('value')

ax1 = fig.add_subplot(222)
ax1.bar([1], [24.26-23], label='SCIV-kNN k=100', bottom=23)
ax1.bar([2], [24.45-23], label='SCIV-kNN k=500', bottom=23)
plt.xticks([0, 1.5, 3], ["", "yoochoose MRR@20", ""], rotation=0)
plt.yticks(np.arange(23, 27, 1), rotation=0)
plt.legend(loc=1)
plt.ylabel('value')

ax1 = fig.add_subplot(223)
ax1.bar([1], [51.47-50], label='SCIV-kNN k=100', bottom=50)
ax1.bar([2], [51.82-50], label='SCIV-kNN k=500', bottom=50)
plt.xticks([0, 1.5, 3], ["", "diginetica recall@20", ""], rotation=0)
plt.yticks(np.arange(50, 54, 1), rotation=0)
plt.legend(loc=1)
plt.ylabel('value')

ax1 = fig.add_subplot(224)
ax1.bar([1], [19.60-18], label='SCIV-kNN k=100', bottom=18)
ax1.bar([2], [19.72-18], label='SCIV-kNN k=500', bottom=18)
plt.xticks([0, 1.5, 3], ["", "diginetica MRR@20", ""], rotation=0)
plt.yticks(np.arange(18, 22, 1), rotation=0)
plt.legend(loc=1)
plt.ylabel('value')
# plt.xlabel(u'指标', FontProperties=font)

# params

# x: 条形图x轴
# y：条形图的高度
# width：条形图的宽度 默认是0.8
# bottom：条形底部的y坐标值 默认是0
# align：center / edge 条形图是否以x轴坐标为中心点或者是以x轴坐标为边缘




plt.show()


