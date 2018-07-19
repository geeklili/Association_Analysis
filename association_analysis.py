import numpy as np
from collections import defaultdict
import pandas as pd


def calculate(data_vector):
    """计算支持度，置信度，提升度
    """
    n_samples, n_features = data_vector.shape
    print('特征数: ', n_features)
    print('样本数: ', n_samples)

    support_dict = defaultdict(float)
    confidence_dict = defaultdict(float)
    lift_dict = defaultdict(float)

    # together_appear: {(0, 1): 3, (0, 3): 2, (0, 4): 1, (1, 0): 3, (1, 3): 2,...}
    # together_appear: 元组里的元素是特征的序号，后面的数字，是这两个特征同时出现的总次数
    together_appear_dict = defaultdict(int)

    # feature_num_dict:{0: 3, 1: 4, 2: 3,...}
    # feature_num_dict: key是特征的序号，后面的数字是这个特征出现的总次数
    feature_num_dict = defaultdict(int)

    # 通过两层的for循环计算特征单独出现的次数，以及两两特征共同出现的次数
    for line in data_vector:
        for i in range(n_features):
            if line[i] == 0:
                continue
            feature_num_dict[i] += 1

            for j in range(n_features):
                if i == j:
                    continue
                if line[j] == 1:
                    together_appear_dict[(i, j)] += 1

    # print(together_appear_dict)
    # print(feature_num_dict)

    # 通过遍历together_appear_dict，计算出两两特征的支持度，置信度，提升度
    for k, v in together_appear_dict.items():
        support_dict[k] = v / n_samples
        confidence_dict[k] = v / feature_num_dict[k[0]]
        lift_dict[k] = v * n_samples / (feature_num_dict[k[0]] * feature_num_dict[k[1]])

    return support_dict, confidence_dict, lift_dict


def create_one_hot(file):
    """将实体数据转换成：0，1数据类型，类似于词袋模型
    """
    with open(file, 'r', encoding='utf-8') as f:
        all_feature_li = []
        content_li = f.readlines()
        line_split_li = [i.strip().split(',') for i in content_li]

        for i in line_split_li:
            for feature in i:
                all_feature_li.append(feature)

        all_feature_set_li = list(set(all_feature_li))
        all_feature_set_li.sort()
        # print(all_feature_set_li)

        feature_dict = defaultdict(int)
        for n, feat in enumerate(all_feature_set_li):
            feature_dict[feat] = n
        # print(feature_dict)

        out_li = list()
        for j in line_split_li:
            feature_num_li = [feature_dict[i] for i in j]
            # print(feature_num_li)
            inner_li = list()
            for num in range(len(all_feature_set_li)):
                if num in feature_num_li:
                    inner_li.append(1)
                else:
                    inner_li.append(0)
            out_li.append(inner_li)

        out_array = np.array(out_li)
        # print(out_array)
        return out_array, feature_dict


def convert_to_sample(feature_dict, s, c, l):
    """把0，1，2，3，... 等字母代表的feature，转换成实体
    """
    # print(feature_dict)
    feature_mirror_dict = dict()
    for k, v in feature_dict.items():
        feature_mirror_dict[v] = k
    # print(feature_mirror_dict)

    support_sample_li = [[feature_mirror_dict[i[0][0]], feature_mirror_dict[i[0][1]], i[1]] for i in s]
    confidence_sample_li = [[feature_mirror_dict[i[0][0]], feature_mirror_dict[i[0][1]], i[1]] for i in c]
    lift_sample_li = [[feature_mirror_dict[i[0][0]], feature_mirror_dict[i[0][1]], i[1]] for i in l]

    return support_sample_li, confidence_sample_li, lift_sample_li


if __name__ == '__main__':
    # 配置路径，如果数据没有经过处理，就配置origin_data_file
    # 如果数据已经经过处理，为0，1数据，就可以直接配置ready_data_file
    origin_data_file = './data/origin.data'
    ready_data_file = './data/sample.data'

    # 如果数据已经构建好了，可以直接读取数组进行计算
    # data = pd.read_csv(ready_data_file)
    # data_array = np.array(data)

    data_array, feature_di = create_one_hot(origin_data_file)
    print(data_array)
    support_di, confidence_di, lift_di = calculate(data_array)
    print('support_di: ', support_di)
    print('confidence_di: ', confidence_di)
    print('lift_di: ', lift_di)

    support = sorted(support_di.items(), key=lambda x: x[1], reverse=True)
    confidence = sorted(confidence_di.items(), key=lambda x: x[1], reverse=True)
    lift = sorted(lift_di.items(), key=lambda x: x[1], reverse=True)
    print('support_li: ', support)
    print('confidence_li: ', confidence)
    print('lift_li: ', lift)

    support_li, confidence_li, lift_li = convert_to_sample(feature_di, support, confidence, lift)
    print('support_li: ', support_li)
    print('confidence_li: ', confidence_li)
    print('lift_li: ', lift_li)
