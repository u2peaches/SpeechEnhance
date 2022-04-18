import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator


if __name__ == "__main__":
    f = open("train_loss/RMSProp.txt")
    x1 = []
    y1 = []
    data= list()
    for line in f:
        data = line.split(",")
        print(data[0][8:-20])
        print(data[3][20:26])
        x1.append(float(data[0][8:-20]))
        y1.append(float(data[3][20:26]))
    f.close()

    f = open("train_loss/RAdam.txt")
    x2 = []
    y2 = []
    data= list()
    for line in f:
        data = line.split(",")
        print(data[0][8:-20])
        print(data[3][20:26])
        x2.append(float(data[0][8:-20]))
        y2.append(float(data[3][20:26]))
    f.close()

    f = open("train_loss/Adabelief.txt")
    x3 = []
    y3 = []
    data= list()
    for line in f:
        data = line.split(",")
        print(data[0][8:-20])
        print(data[3][20:26])
        x3.append(float(data[0].split(":")[1][0:-20]))
        y3.append(float(data[3][20:26]))
    f.close()

    x_major_locator = MultipleLocator(50)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(10)
    # 把y轴的刻度间隔设置为10，并存在变量里

    # 把y轴的主刻度设置为10的倍数

    plt.plot(x1, y1, 'r-', label=u'RMSProp')
    plt.plot(x2, y2, 'b-', label=u'RAdam')
    plt.plot(x3, y3, 'g-', label=u'Adabelief')
    plt.legend()


    plt.xlabel(u'iter')
    plt.ylabel(u'loss')
    plt.title('Compare loss for different models in training.')
    # fig = plt.figure(figsize=(7, 5))
    # ax1 = fig.add_subplot(1, 1, 1)  # ax1是子图的名字
    ax1 = plt.gca()
    # ax为两条坐标轴的实例
    ax1.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax1.yaxis.set_major_locator(y_major_locator)
    # "g"代表green，表示画出的曲线是绿色，"-"表示画出的曲线是实线，label表示图例的名称
    #y1.reverse()

    plt.xlim(0, 470)
    plt.ylim(0, 100)
    plt.savefig("train_results_loss.png")
    plt.show()
