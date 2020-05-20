import matplotlib.pyplot as plt



#折线图
x = [1, 3, 5, 7, 9, 11, 15, 21]
L1 = [3.7, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.5]
L2 = [3.1, 2.8, 3.1, 3., 3.3, 3.3, 3.6, 3.7]
plt.plot(x,L1,'s-',color = 'r',label="L1")#s-:方形
plt.plot(x,L2,'o-',color = 'g',label="L2")#o-:圆形
plt.xlabel("k")#横坐标名字
plt.ylabel("misclassification rate/%")#纵坐标名字
plt.legend(loc = "best")#图例
plt.grid()
plt.show()
