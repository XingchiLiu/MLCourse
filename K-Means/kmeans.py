from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# 如果存在文件夹，删除文件夹
if os.path.exists("./img"):
    # 不管有没有内容，都删
    shutil.rmtree("./img")
os.mkdir("./img")


def get_sample_with_center(k):
    """
    获得有k个中心的随机的簇
    :param k:
    :return:
    """
    # 获得k个随机的中心
    random_center = np.random.randn(k, 2)*3 + [0, 0]

    data = np.random.randn(100 % k, 2)*3 + [0, 0]
    for i in range(k):
        new_cluster = np.random.randn(100//k, 2) + random_center[i]
        data = np.concatenate((data, new_cluster), axis=0)

    return data


def gen_center(X_train, k):

    n_feature = X_train.shape[1]

    # 样本平均值，纵向相加[x_mean, y_mean]
    f_mean = np.mean(X_train, axis=0).reshape((1, n_feature))
    # 获取标准差
    f_std = np.std(X_train, axis=0).reshape((1, n_feature))

    # 获得3*2的标准正太分布的随机数组，其中心是f_mean，
    centers = np.random.randn(k, n_feature)*f_std + f_mean    # (k,n_feature)
    return centers


def k_means(X_train, centers):

    # 样本数量
    n_sample = X_train.shape[0]
    dist = np.zeros((n_sample, k))  # 每个样本对每个质心都有一个距离

    cent_pre = np.zeros(centers.shape)
    cent_cur = gen_center(X_train, k)
    cent_move = np.linalg.norm(cent_cur - cent_pre)  # 每轮迭代后质心的移动距离

    epsilon = 1e-3  # 质心需要移动的最小距离
    epoch = 0  # 当前迭代次数
    max_iter = 50  # 最大迭代次数

    fig = plt.figure(1)
    # 开启交互模式，图片可以动态变化
    plt.ion()
    while epoch < max_iter and cent_move > epsilon:
        epoch += 1

        # 首先计算每个样本离每个质心的距离
        for i in range(k):
            dist[:, i] = np.linalg.norm(X_train - cent_cur[i], axis=1)

        # 样本对应的类别为距离最近的质心
        clusters = np.argmin(dist, axis=1)

        cent_pre = deepcopy(cent_cur)

        # 计算每个类别下的均值坐标，更新质心
        for i in range(k):
            cent_cur[i] = np.mean(X_train[clusters == i], axis=0)

        # 更新中心移动距离
        cent_move = np.linalg.norm(cent_cur - cent_pre)

        # 画图
        plt.clf()
        plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.5, c=clusters)
        plt.scatter(cent_cur[:, 0], cent_cur[:, 1], marker='*', c='k')
        # 保存图片
        ffpath = "./img/" + str(epoch) + ".png"
        plt.savefig(ffpath)

        plt.show()
        plt.pause(1)

    plt.ioff()


if __name__ == '__main__':
    k = 3
    # 随机的100个样本和3个随机的点
    X_train = get_sample_with_center(k)
    centers = gen_center(X_train, k)
    k_means(X_train, centers)
