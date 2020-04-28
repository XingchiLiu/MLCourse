import pandas as pd
from math import log
import operator
from DecisionTree import tree_plotter


def get_data_set(filepath):
    """
    从文件读取训练集
    :param filepath:  文件名
    :return:  一个dataframe
    """
    f = open(filepath)
    line_list = f.read().strip().split('\n')
    data = []
    for line in line_list:
        words = line.split(maxsplit=4)
        data.append(words)
    columns = ['age', 'prescript', 'astigmatic', 'tearRate', 'type']
    df = pd.DataFrame(data=data, columns=columns)
    # print(df)
    return df


def shannon_entropy(df:pd.DataFrame):
    """
    根据一个dataframe获取信息熵
    :param df:
    :return:
    """
    first_column = df.columns[0]
    labels = df.groupby('type')[first_column].count()
    # print(labels)
    n_types = len(labels)
    type_count = labels.values
    shannon_ent = 0.0
    for i in range(n_types):
        prob = float(type_count[i]) / n_types
        shannon_ent = shannon_ent - prob * log(prob, 2)
    return shannon_ent


def split_data_set(target: pd.DataFrame, feature: str, value: str) -> pd.DataFrame:
    """

    :param target: 需要划分的数据集
    :param feature: 根据哪个特征来划分
    :param value: 该特征的哪个取值
    :return:  经过划分的数据集，注意，会去掉该特征
    """
    # 根据feature进行筛选，并且去掉一列
    result = target[target[feature] == value].drop(feature, axis=1)
    return result


def find_feature_to_split(target: pd.DataFrame) -> str:
    """

    :param target: 需要划分的数据集
    :return: 区分度最大的特征
    """
    features = target.columns[:-1]
    original_entropy = shannon_entropy(target)

    max_info_gain = 0
    # 默认的最佳特征，可能没有信息增益
    best_feature = features[-1]

    for feature in features:
        # 计算该特征下的信息熵
        unique_feature_val_list = set(target[feature])
        new_entropy = 0
        for feature_val in unique_feature_val_list:
            sub_data_set = split_data_set(target, feature, feature_val)
            prob = len(sub_data_set) / float(len(target))
            new_entropy += prob * shannon_entropy(sub_data_set)
        info_gain = original_entropy - new_entropy  # 计算信息增益，g(D,A)=H(D)-H(D|A)
        # 最大信息增益
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature

    return best_feature


def create_tree(target: pd.DataFrame):
    """
    构建决策树
    :param target: 数据集
    :return: 一棵树，形式如{feature1:{f1_v1:{}, f1_v2:{}}}
    """

    class_list = list(target['type'])   # 包含类别的df
    # 类别相同，停止划分
    if class_list.count(class_list[-1]) == len(class_list):
        return class_list[-1]

    # 按照信息增益最高选取分类特征属性
    best_feature = find_feature_to_split(target)  # 返回分类的特征

    my_tree = {best_feature: {}}  # 构建树
    feature_values = target[best_feature]
    unique_feature_values = set(feature_values)
    for feature_value in unique_feature_values:
        # 构建数据的子集合，并进行递归
        sub_data_set = split_data_set(target, best_feature, feature_value)  # 待划分的子数据集
        my_tree[best_feature][feature_value] = create_tree(sub_data_set)
    return my_tree


df = get_data_set('./lenses.txt')
decision_tree = create_tree(df)
print("决策树：", decision_tree)
tree_plotter.create_plot(decision_tree)

