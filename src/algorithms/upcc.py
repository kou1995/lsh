# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:46:32 2018
UPCC算法实现
@author: Yanchao
"""
import numpy as np
import h5py
from util.dataset_util import load_data_from_ws_dream, \
    erase_data

class UPCCRecommender():
    def __init__(self, raw_data, data, top=30):
        self.raw_data = raw_data
        self.data = data
        self.top = top
        (self.num_of_users, self.num_of_services) = data.shape
        self.pearson_similarity_matrix = None

    def calculate_pearson_similarity_matrix(self):
        """
        This method is similar with classify method in basiculsh.py
        both of which calculate the pearson similarity matrix of users
        To save time, the similarity matrix is calculated by matrix operation
        :return:
        """
        user_average_data = np.average(self.data, axis=1).reshape(self.num_of_users, 1)
        variance = self.data - user_average_data

        covariance_matrix = np.dot(variance, variance.T)

        square_variance = np.sum(np.square(variance), axis=1).reshape(self.num_of_users, 1)
        delta = np.sqrt(np.dot(square_variance, square_variance.T))

        self.similarity_matrix = covariance_matrix/delta
        return self.similarity_matrix

    def classify(self):
        """
        This method is similar with classify method in basiculsh.py
        both of which calculate the cosine similarity matrix of users
        To save time, the similarity matrix is calculated by matrix operation
        :return:
        """
        norms = np.linalg.norm(self.data, axis=1)
        norms = norms.reshape(self.data.shape[0], 1)
        norms_matrix = np.dot(norms, norms.T)

        dot_matrix = np.dot(self.data, self.data.T)

        self.similarity_matrix = dot_matrix / norms_matrix

        for i in range(self.num_of_users):
            self.similarity_matrix[i, i] = 0

        return self.similarity_matrix

    def findSimilarUsers(self, index, top=30):
        similarities = np.zeros(self.num_of_users)
        for i in range(self.num_of_users):
            if i == index:
                similarities[i] = 0
            else:
                similarities[i] = self.calculateSimilarity(index, i)

        return np.argsort(similarities)[-top:], np.sort(similarities)[-top:].reshape(1, top)

    def calculateSimilarity(self, u1, u2):
        # 两个向量相乘，结果为1的元素即是二者的交集
        # product = self.data[u1, :] * self.data[u2, :]
        # user1 = self.data[u1, product > 0]
        # user2 = self.data[u2, product > 0]
        user1 = self.data[u1, :]
        user2 = self.data[u2, :]

        if len(user1) == 0 or len(user2) == 0:
            return 0

        avg_user1 = np.average(user1)
        avg_user2 = np.average(user2)

        numerator = np.dot(user1 - avg_user1, user2 - avg_user2)
        denominator = np.sqrt(np.dot(user1 - avg_user1, user1 - avg_user1)) * \
                      np.sqrt(np.dot(user2 - avg_user2, user2 - avg_user2))

        if denominator > 0:
            return numerator / denominator
        else:
            return 0

    def evaluate(self, indices):
        rmse = 0
        mae = 0
        num_of_similar = 0
        num_of_samples = 0
        valid_similar_users = 0
        num_of_predicted = 0
        num_of_failure = 0
        #首先计算每个用户的平均值
        data_copy = np.copy(self.data)
        valid_flags = self.data > 0
        data_copy[data_copy==-1] = 0
        user_avg_data = np.sum(data_copy, axis=1)/np.sum(valid_flags, axis=1)
        user_avg_data = user_avg_data.reshape((self.num_of_users, 1))

        for idx in indices:
            # 找出待预测的列号，如果需要预测值为1，否则为0
            predicted_columns = np.multiply(self.data[idx] == 0, self.raw_data[idx]>0)
            # 提取出相似用户的所有数据
            similar_users, similarities = self.findSimilarUsers(idx, self.top)
            data_for_predicted = data_copy[similar_users,:]

            # 提取出相似用户的数据中参与预测的列
            data_for_predicted = data_for_predicted[:, predicted_columns>0]
            data_for_predicted_valid_flag = data_for_predicted > 0
            data_for_reference = self.raw_data[idx, predicted_columns>0]
            # 计算各列有效值（> 0）的个数
            valid_counts = np.dot(similarities, data_for_predicted_valid_flag).squeeze()
            # 这里做了一个优化，无法预测的列不应该直接舍弃，应该给他置平均值

            predicted_values = np.zeros((1, data_for_predicted.shape[1]))
            #如果没有可预测的列，置所有待预测列的值为用户的平均值
            if (np.sum(valid_counts) == 0):
                predicted_values[:, :] = user_avg_data[idx]
            else:
                predicted_values[:, valid_counts == 0] = user_avg_data[idx]
                valid_counts[valid_counts == 0] = 1
                predicted_values = np.dot(similarities, data_for_predicted * data_for_predicted_valid_flag) \
                               / valid_counts

            predicted_values = predicted_values.squeeze()

            #以前的方法只计算推荐的服务的mae，这个结果不能客观地评价算法
            # 计算所有的预测值的mae和rmse
            mae += np.sum(np.abs(predicted_values - data_for_reference))
            rmse += np.dot(predicted_values - data_for_reference, predicted_values - data_for_reference)
            num_of_predicted += data_for_reference.shape[0]

        return mae/num_of_predicted, np.sqrt(rmse/num_of_predicted)

if __name__ == '__main__':
    def tune_param(ratio = 0.1, times=50):
        """
        using ws-dream dataset, throughput data
        sparsity ratio: 10%
        :return: b (number of hash tables), r (number of hash function)
        """
        raw_data = load_data_from_ws_dream(1)

        top_options = [3, 5, 10, 30, 50]

        maes = np.zeros((times, len(top_options)))
        rmses = np.zeros_like(maes)

        num_of_test_samples = 50

        for t in range(times):
            seed = t + 1
            data = erase_data(raw_data, ratio, seed=seed)
            np.random.seed(seed)
            test_data = np.random.choice(data.shape[0], num_of_test_samples, replace=False)

            for (idx, top) in enumerate(top_options):
                recommender = UPCCRecommender(raw_data, data, top=top)
                recommender.classify()

                maes[t, idx], rmses[t, idx] = \
                    recommender.evaluate(test_data)

            if (t + 1) % 10 == 0:
                print('>')
            else:
                print('>', end='')

        # f = h5py.File('../../output/ulsh.h5py', 'a')
        # f.create_group('/upcc');
        # f.create_dataset('/upcc/maes', data=maes)
        # f.create_dataset('/upcc/rmses', data=rmses)
        #
        # f.close()
        #
        # return np.argmin(np.average(maes, axis=0)),\
        #        np.argmin(np.average(rmses, axis=0))

    tune_param(times=1)
