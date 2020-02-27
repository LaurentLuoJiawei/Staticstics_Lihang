# -*- coding: utf-8 -*-
import numpy as np
import random
import time
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import accuracy_score

pwd = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(pwd,'result.txt')
Learning_rate = 0.0001                                 # 学习步长
MAX_iter = 10000                                 # 学习次数
feature_length = 324                                # hog特征维度
classes = 2 #类别


def get_hog_features(trainset):
    """
    版权声明：本文为CSDN博主「wds2006sdo」的原创文章，遵循
    CC
    4.0
    BY - SA
    版权协议，转载请附上原文出处链接及本声明。
    原文链接：https: // blog.csdn.net / wds2006sdo / article / details / 51923546
    :param trainset:
    :return:
    """
    features = []

    hog = cv2.HOGDescriptor('../hog.xml')

    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features, (-1, 324))

    return features



class Perception(object):
    def __init__(self,num_features,learning_rate):
        self.num_features = num_features
        self.a = np.zeros(self.num_features).reshape(1,-1) # (1,num_features)
        self.b = np.zeros(1) # int
        self.learning_rate = learning_rate
        self.w = np.zeros(self.num_features+1).reshape(1, -1)  # (1,num_features)

    def train(self, X, Y):
        """
        对偶形式的训练：
            1。 构建gram 矩阵，若样本为二维，则矩阵维度(样本数量，特征数量，特征数量)
                对于每一行的矩阵为特征之间的相乘，[x1*x1,x1*2,...,x1*xn]，这样更新梯度的时候直接查表。
            2. 权重更新:
                第i 行，取出该行的数据，a[1,num_features] * xi.t[num_fetures]
            3。 判断是否误分类
            4。 更新a，所以a，b 的数据类型要初始化为list 便于权重更新，但是矩阵计算可以考虑用numpy

        :param x: numpy.array (num_samples, num_features)
        :param y: ;list [int,]
        :return: self.a,self.b
        """
        correct_count = 0
        times = 0
        MAX_ITER = 5000
        num_samples = len(X)
        # 异常判断
        assert len(X) == len(Y)
        num_features = X.shape[1]
        # for 循环得到矩阵
        t1 = time.time()
        gram_matrix_1 = np.zeros(num_samples * num_features * num_features).reshape(
            [num_samples, num_features, num_features])
        for i in range(num_samples):
            line = X[i, :]
            line = line[:, np.newaxis]
            gram_matrix_1[i] = line.dot(line.transpose(1, 0))
        print("for loop gram_matrix use %s" % (str(time.time() - t1)))
        # 直接通过矩阵计算得到矩阵
        # t2 = time.time()
        # temp_X = X[:, :, np.newaxis]
        # gram_matrix_2 = np.matmul(temp_X, temp_X.transpose(0, 2,1))
        # print("for matrix_mul gram_matrix use %s" % (str(time.time() - t2)))

        # 由书本公式证明，只要数据集线性可分，那么就一定能在有限次数内找到超平面
        """
            误分类则进行梯度更新
        """
        while times <= MAX_ITER and correct_count <= MAX_ITER:
            index = random.choice(range(len(X)))
            line = gram_matrix_1[index] # (num_features,num_features)
            y = Y[index]
            if ((y * np.sum(np.dot(self.a,line))) ) * y < 0:
                self.a += self.a + self.learning_rate
                self.b += self.b + self.learning_rate*y
            else:
                correct_count += 1
            times += 1
        print("train completed, iterations= %d" %times)

    def train_orig(self, X, Y):
        pass

    def predict_one(self,x):
        """

        :param new_x: numpy.array(1,num_features)
        :return: y
        """
        res = sum(np.dot(self.a, x)) + self.b
        return res

    def predict(self, X):
        res = [(np.dot(self.a, i.transpose(1, 0)) + self.b) for i in X]
        return res



def performance(test_set, Percetion):
    pass


def train_progress_plot(x, loss):
    pass

if __name__ == '__main__':
    """
    先对函数进行测试
    """
    print("start and initialize the Percetion")
    perception = Perception(num_features=feature_length,learning_rate=Learning_rate)
    t1 = time.time()
    print("load data")
    raw_data = pd.read_csv(os.path.join(pwd,'data/train_binary.csv'), header=0)
    data = raw_data.values
    imgs = data[0::, 1::]
    labels = data[::, 0]
    features = get_hog_features(imgs)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,random_state=1)
    print(" use %s to load the data"%(str(time.time()-t1)))
    print("start training ")
    t2 = time.time()
    perception.train(X=train_features,Y=train_labels)
    print("cost ",t2-time.time(),"s to train")
    print("start test")
    res = perception.predict(test_features)
    acc = accuracy_score(res,test_features)
    print("accuracy score is ",acc)
    """
    测试用例
    """
    # perception = Perception(num_features=60,learning_rate=Learning_rate)
    # X = np.array([i for i in range(600)]).reshape([10,60])
    # Y = [1,-1,1,1,-1,1,1,-1,1,1]
    # print(perception.train(X,Y))


