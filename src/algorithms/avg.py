import numpy as np
import logging

class UAVGRecommender:
    """

    """
    def __init__(self, raw_data, data, seed=1):
        self.raw_data = raw_data
        self.data = data

        (self.num_of_users, self.num_of_services) = data.shape


    def evaluate(self, indices, num=-1):
        """
        :param indices:
        :param num: the number of users random choosed from all users
        :return:
        """
        rmse = 0
        mae = 0
        num_of_similar = 0
        num_of_samples = 0
        num_of_predicted = 0
        num_of_valid_similar = 0
        user_indices = np.arange(0, self.num_of_users)
        for idx in indices:
            #找出待预测的列号，如果需要预测值为1，否则为0
            predicted_columns = np.multiply(self.data[idx] == 0, self.raw_data[idx]>0)
            # random chooosing users to predict the target user's value
            candidates = np.arange(self.num_of_users)
            candidates = candidates[candidates != idx]
            if num > 0:
                ref_users = np.random.choice(candidates, num, replace=False)
            else:
                ref_users = candidates

            #提取出相似用户的所有数据
            data_for_predicted = self.data[ref_users]
            num_of_similar += data_for_predicted.shape[0]
            #提取出相似用户的数据中参与预测的列
            data_for_predicted = data_for_predicted[:, predicted_columns>0]
            data_for_reference = self.raw_data[idx, predicted_columns>0]
            #计算各列有效值（> 0）的个数
            valid_counts = np.sum(data_for_predicted>0, axis=0)
            #去除有效值个数为0的列
            data_for_predicted = data_for_predicted[:, valid_counts>0]
            # 如果有效的列，中止本次循环
            if (data_for_predicted.shape[1] == 0):
                continue
            data_for_reference = data_for_reference[valid_counts>0]
            valid_counts = valid_counts[valid_counts>0]
            #将data_for_predicted中的-1置为0
            data_for_predicted[data_for_predicted==-1] = 0
            #预测各列的值
            rt = np.sum(data_for_predicted, axis=0)/valid_counts;
            #找出预测值最小的列号
            min_index = rt.argmin()

            #计算所有的预测值的mae和rmse
            mae += np.sum(np.abs(rt - data_for_reference))
            rmse += np.dot(rt-data_for_reference, rt-data_for_reference)
            num_of_predicted += data_for_reference.shape[0]

        return mae/num_of_predicted, np.sqrt(rmse/num_of_predicted)

            #以前的mae和rmse的计算方式，这是有问题的
        #     # error = np.abs(rt[min_index] - data_for_reference[min_index])
        #     mae += error
        #     rmse += error * error
        #     num_of_samples += 1
        #     num_of_valid_similar += valid_counts[min_index]
        #
        # return mae/num_of_samples, np.sqrt(rmse/num_of_samples), num_of_similar/num_of_samples, num_of_valid_similar/num_of_samples





