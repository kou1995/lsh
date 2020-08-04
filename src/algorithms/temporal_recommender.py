import numpy as np
from core.lsh import LSH
from util.dataset_util import load_temporal_data, erase_data
import h5py
import time

class TemporalRecommender:
    def __init__(self, raw_data, data, num_of_functions=2, num_of_tables=10, seed=1):
        '''
        :param data: shape (num_of_users, num_of_services, num_of_time_slices)
        :param num_of_functions:
        :param num_of_tables:
        :param seed: 随机函数的种子
        '''
        self.raw_data = raw_data
        self.data = data
        (self.num_of_users, self.num_of_services, self.num_of_slices) = data.shape
        # initialize lsh tables
        self.num_of_tables = num_of_tables
        self.lsh_family = []
        self.seed = seed

        for i in range(self.num_of_tables):
            self.lsh_family.append(LSH(num_of_functions))
            self.lsh_family[i].fit(self.num_of_services, seed)
            seed += 1

    def classify(self):
        '''
        compute lsh value of items, and put them into different buckets by its lsh value
        :return:
        '''
        self.similarity_matrix = np.zeros((self.num_of_users, self.num_of_slices, self.num_of_slices))
        for i in range(self.num_of_tables):
            for j in range(self.num_of_users):
                hash_values = self.lsh_family[i].get_batch_hash_value(self.data[j, :, :])
                for k in range(self.num_of_slices):
                    self.similarity_matrix[j, k, :][hash_values == hash_values[k]] = 1

        # 1 第一次测试，定义两个用户的相似度为前63个时刻两用户相似度的平均值
        # self.similarity_matrix = np.average(self.similarity_matrix, axis=2)

    def evaluate_old(self, test_data, threshold=0):
        '''
        预测test_data中值为0的response time值，并计算所有预测值的绝对差
        evaluate mae
        :param test_data:
        :param reference_data:
        :param threshold: 找相似用户的阈值，默认为0， 与传统的LSH方法一致
        :return:
        '''

        # 找出每个用户的所有sevice中预测值最小的，推荐给用户，并计算其mae
        bias = []
        num_of_similars = []
        num_of_fails = 0
        num_of_predicted = 0
        for i in range(self.num_of_users):
            user = test_data[i, :]
            similar_slices = np.argwhere(self.similarity_matrix[i, self.num_of_slices - 1, :] > threshold)

            if len(similar_slices) == 0:
                print('user %d has no similar slices!' % (i))
                continue

            indices = np.argwhere(user == 0)
            min_index = -1
            min_rt = np.iinfo(np.int32).max
            for j in indices:
                if self.data[i, j, self.num_of_slices - 1] != -1:
                    similar_slices_values = self.data[i, j, :][:, similar_slices]
                    valid_similar_slices_values = similar_slices_values[similar_slices_values > 0]

                    num_of_similar = len(valid_similar_slices_values)
                    num_of_similars.append(num_of_similar)

                    if num_of_similar > 0:
                        predicted_value = np.average(valid_similar_slices_values)
                        if predicted_value < min_rt:
                            min_rt = predicted_value
                            min_index = j
                    else:
                        num_of_fails += 1

            if min_index != -1:
                #     print('user %d predict failed, num of predicted is %i' % (i, len(indices)))
                # else:
                bias.append(min_rt - self.data[i, min_index, self.num_of_slices - 1])
                num_of_predicted += 1

        bias = np.squeeze(bias)

        rmae = np.sqrt(np.dot(bias, bias) / num_of_predicted)

        return rmae, np.average(num_of_similars), num_of_predicted  # ,

    def evaluate(self, indices, t):
        # 获取所有测试目标用户的相似矩阵行
        similar_slices_mat = self.similarity_matrix[indices, t, :]
        # part_similarity_matrix = self.similarity_matrix[indices, :]
        #
        # #该代码实现的是将与目标用户i最相似的top用户对应的索引置为1
        # for i in range(len(indices)):
        #     similar_user_mat[i, part_similarity_matrix[i, self.part_similarity_matrix[i, :] > threshold]] = 1

        # shape: (len(indices, n)
        # if predict_columns[i, j] = 1, it means we should predict the quality of sevice j invoked by user i
        predict_columns = np.multiply(self.data[indices, :, t] == 0, self.raw_data[indices, :, t] > 0).astype(float)

        # shape(len(indices), self.num_of_services)
        # valid_counts[i, j] = 1 represents the quality value when user i invoke service j \
        # can be predicted by their similar users' quality value of invoking service j \
        # but it doesn't mean the quality value have to be predicted
        valid_counts = np.zeros((len(indices), self.num_of_services))
        for (idx, target) in enumerate(indices):
            valid_counts[idx, :] = np.dot(similar_slices_mat[idx, :], (self.data[target, :, :] > 0).astype(float).T)

        # set values that can't be predicted to the average value of the target user's qos value.
        # compute the average value of the qos value when user i invokes sevice j
        # shape of avg_value of user: (len(indices), self.num_of_services)
        avg_value_of_user = np.average(self.data[indices, :, :], axis=2)
        # If a user has no valid qos values, set it's average value to the average value of the whole matrix
        avg_value_of_user[avg_value_of_user == 0] = np.average(self.data)

        # to avoid divide by 0, set 0 to 1
        valid_counts[valid_counts == 0] = 1

        # shape:(len(indices), n)

        predicted_value = np.zeros((len(indices), self.num_of_services))
        for (idx, target) in enumerate(indices):
            data_for_predicted = self.data[target, : , :].copy()
            data_for_predicted[data_for_predicted == -1] = 0
            predicted_value[idx, :] = np.dot(similar_slices_mat[idx].reshape(1, self.num_of_slices), data_for_predicted.T)
            predicted_value[idx, predicted_value[idx] == 0] = avg_value_of_user[idx, predicted_value[idx] == 0]
            predicted_value[idx, :] = np.multiply(predict_columns[idx], predicted_value[idx])
            predicted_value[idx, :] = np.divide(predicted_value[idx], valid_counts[idx])

        total_predicted_count = np.sum(predicted_value > 0)
        # 计算所有预测值的MAE和RMSE
        reference_data = self.raw_data[indices, :, t]
        reference_data[predicted_value == 0] = 0
        mae = np.sum(np.abs(predicted_value - reference_data)) / total_predicted_count
        rmse = np.sum((predicted_value - reference_data) * (predicted_value - reference_data))
        rmse = np.sqrt(rmse / total_predicted_count)

        return mae, rmse, np.average(similar_slices_mat)

if __name__ == '__main__':
    def tune_param(ratio = 0.1, times = 50):
        """
        using ws-dream dataset, throughput data
        sparsity ratio: 10%
        :return: b (number of hash tables), r (number of hash function)
        """
        raw_data = load_temporal_data('rtdata')

        # normalized_raw_data = preprocessing.scale(raw_data)
        num_of_hash_functions = [4]#, 6, 8, 10]
        num_of_hash_tables = [8]#, 6, 4, 2]
        col_num = len(num_of_hash_tables)
        # 通过测试发现，10*4, 10*6, 10*8, 10*10, 8*4, ...., 4*4, 4*6, 4*8, 4*10 (*前面为hash_table的个数，后面为hash_function的个数)
        # 上述参数设置顺序下返回的相似用户数呈递减的趋势，因此我们索性将hash_table和hash_function的数目组合看作一个一维的结构
        total = len(num_of_hash_tables) * len(num_of_hash_functions)
        maes = np.zeros((times, len(num_of_hash_tables), len(num_of_hash_functions)))
        rmses = np.zeros_like(maes)
        num_of_similars = np.zeros_like(maes)

        num_of_test_samples = 10

        for t in range(times):
            seed = t + 1
            data = erase_data(raw_data, ratio, seed=seed)
            np.random.seed(seed)
            test_data = np.random.choice(data.shape[0], num_of_test_samples, replace=False)

            for (t_idx, table_num) in enumerate(num_of_hash_tables):
                for (f_idx, function_num) in enumerate(num_of_hash_functions):
                    recommender = TemporalRecommender(raw_data, data, num_of_tables=table_num,
                                                        num_of_functions=function_num, seed=seed)
                    recommender.classify()

                    maes[t, t_idx, f_idx], rmses[t, t_idx, f_idx], num_of_similars[t, t_idx, f_idx] = \
                        recommender.evaluate(test_data, 63)

            if (t + 1) % 10 == 0:
                print('>')
            else:
                print('>', end='')

        print(maes)

        # f = h5py.File('../../output/ulsh.h5py', 'a')
        # f.create_group('/baisc_ulsh');
        # f.create_dataset('/basic_ulsh/maes', data=maes)
        # f.create_dataset('/basic_ulsh/rmses', data=rmses)
        # f.create_dataset('/basic_ulsh/num_of_similars', data=num_of_similars)
        #
        # f.close()

        # return np.argmin(np.average(maes.reshape(times, 16), axis=0)), \
        #        np.argmin(np.average(num_of_similars.reshape(times, 16), axis=0))

    begin = time.time()
    tune_param(times=1)
    print(time.time() - begin)
