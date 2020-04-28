import matplotlib.pyplot as plt

"""
这里建立一个数组。表示连续index抽没有出6星，然后这一抽抽出6星的概率，index从0开始
p_6star = [2%, 2%, ...., 2%, 4%, 6%, ..., 100%]
所以这是个长为99的数组，index=98时，p_6star[index]=100%
"""
p_6star = [0.0] * 99
for index in range(99):
    if index < 50:
        p_6star[index] = 0.02
    else:
        p_6star[index] = 0.02 + (index - 49) * 0.02

'''
我们现在一共有100个状态
其中99个是没有抽中w的状态，1个是已经抽中w的状态
'''
p_no_w = [0.0] * 99
p_w = 0.0
p_no_w[0] = 1.0


def draw_card():
    global p_w
    global p_no_w

    # 根据概率抽卡，更新状态
    p_not_w_new = [0.0] * 99
    # 计算p_not_w_new[0]
    for i in range(99):
        # 在这些任意一个阶段抽中非w的6星都会重新回到状态0
        p_not_w_new[0] += p_no_w[i] * p_6star[i] * 0.65
        p_w += p_no_w[i] * p_6star[i] * 0.35
    # 计算p_not_w_new[1-98]
    for i in range(1, 99):
        # 只有一种可能，那就是上一次没有抽中6星
        p_not_w_new[i] = p_no_w[i - 1] * (1 - p_6star[i - 1])

    p_no_w = p_not_w_new

    # 后置条件
    assert abs((sum(p_no_w) + p_w) - 1) < 0.000001


# 列表，表示第index抽以后已经抽中了w的概率
p_get_w = [0.0] * 300
for i in range(300):
    draw_card()
    p_get_w[i] = p_w

p_get_w[299] = 1.0

for i in range(300):
    print(str(i+1) + ' :%.1f%%'%(p_get_w[i]*100))

p_get_w_no_guarantee = [1 - (1 - 0.007) ** i for i in range(300)]


# 可视化
plt.plot(range(300), p_get_w)
x_show = [1, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]
for x, y in zip(x_show, [p_get_w[i - 1] for i in x_show]):
    plt.text(x - 1, y, '(%d,%.1f%%)' % (x, y*100), ha='center', va='bottom', fontsize=10)
plt.plot(range(300), p_get_w_no_guarantee)
plt.rcParams['font.sans-serif'] = 'simhei'
plt.legend(['有软保底和硬保底', '无任何保底'], loc='lower right', fontsize=10)
plt.show()
