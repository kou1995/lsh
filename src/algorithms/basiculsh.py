import numpy as np
from util.dataset_util import load_data_from_ws_dream, \
    erase_data
from core.lsh import LSH
from algorithms.lsh_recommender import LSHRecommender
import h5py

class ULSHRecommender(LSHRecommender):
    """
    This class implemented a user-based CF method, which use local sensitive hash method to calculate
    the similarity of users. This method can preserve the privacy of users.
    """
    def __init__(self, raw_data, data, num_of_functions = 8, num_of_tables = 4, seed = 1):
        self.raw_data = raw_data
        self.data = data

        (self.num_of_users, self.num_of_services) = data.shape
        self.num_of_tables = num_of_tables
        self.num_of_functions = num_of_functions
        # initializing users' similarity matrix
        self.similarity_matrix = np.zeros((self.num_of_users, self.num_of_users))
        self.lsh_family = []
        for i in range(num_of_tables):
            self.lsh_family.append(LSH(num_of_functions))
            self.lsh_family[i].fit(self.num_of_services, seed + i)


    def classify(self):
        """"
        calculate the users' similarity matrix
        element m[i,j] in matrix represents how many times
        user i and user j have same hash functions in the hash table
        :return:
        """

        for i in range(self.num_of_tables):
            bucket = {}
            # shape of hash_values: (1, self.num_of_users)
            hash_values = self.lsh_family[i].get_batch_hash_value(self.data.T)
            for j in range(self.num_of_users):
                self.similarity_matrix[j, :][hash_values == hash_values[j]] = 1
        #置相似度矩阵对角线中的元素全为0，即用户和它本身不相似，这是为了以后处理方便
        for i in range(self.num_of_users):
            self.similarity_matrix[i, i] = 0

        return self.similarity_matrix

if __name__ == '__main__':
    def tune_param(ratio = 0.1, times = 50):
        """
        using ws-dream dataset, throughput data
        sparsity ratio: 10%
        :return: b (number of hash tables), r (number of hash function)
        """
        raw_data = load_data_from_ws_dream(1)

        # normalized_raw_data = preprocessing.scale(raw_data)
        num_of_hash_functions = [4, 8, 12, 16]
        num_of_hash_tables = [10, 8, 6, 4]
        col_num = len(num_of_hash_tables)
        # 通过测试发现，10*4, 10*6, 10*8, 10*10, 8*4, ...., 4*4, 4*6, 4*8, 4*10 (*前面为hash_table的个数，后面为hash_function的个数)
        # 上述参数设置顺序下返回的相似用户数呈递减的趋势，因此我们索性将hash_table和hash_function的数目组合看作一个一维的结构
        total = len(num_of_hash_tables) * len(num_of_hash_functions)
        maes = np.zeros((times, len(num_of_hash_tables), len(num_of_hash_functions)))
        rmses = np.zeros_like(maes)
        num_of_similars = np.zeros_like(maes)

        num_of_test_samples = 50

        for t in range(times):
            seed = t + 1
            data = erase_data(raw_data, ratio, seed=seed)
            np.random.seed(seed)
            test_data = np.random.choice(data.shape[0], num_of_test_samples, replace=False)

            for (t_idx, table_num) in enumerate(num_of_hash_tables):
                for (f_idx, function_num) in enumerate(num_of_hash_functions):
                    recommender = ULSHRecommender(raw_data, data, num_of_tables=table_num,
                                                        num_of_functions=function_num, seed=seed)
                    recommender.classify()

                    maes[t, t_idx, f_idx], rmses[t, t_idx, f_idx], num_of_similars[t, t_idx, f_idx] = \
                        recommender.evaluate(test_data)

            if (t + 1) % 10 == 0:
                print('>')
            else:
                print('>', end='')

        # f = h5py.File('../../output/ulsh.h5py', 'a')
        # f.create_group('/baisc_ulsh');
        # f.create_dataset('/basic_ulsh/maes', data=maes)
        # f.create_dataset('/basic_ulsh/rmses', data=rmses)
        # f.create_dataset('/basic_ulsh/num_of_similars', data=num_of_similars)
        #
        # f.close()
        #
        # return np.argmin(np.average(maes.reshape(times, 16), axis=0)), \
        #        np.argmin(np.average(num_of_similars.reshape(times, 16), axis=0))

    tune_param(times=1)



