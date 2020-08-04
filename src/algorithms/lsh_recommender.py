import numpy as np

class LSHRecommender:
    def __init__(self, raw_data, data, num_of_hash_function, num_of_hash_table):
        pass

    def evaluate(self, indices, threshold = 0):
        #获取所有测试目标用户的相似矩阵行
        similar_user_mat = np.zeros((len(indices), self.num_of_users))
        similar_user_mat[self.similarity_matrix[indices, :] > threshold] = 1
        # part_similarity_matrix = self.similarity_matrix[indices, :]
        #
        # #该代码实现的是将与目标用户i最相似的top用户对应的索引置为1
        # for i in range(len(indices)):
        #     similar_user_mat[i, part_similarity_matrix[i, self.part_similarity_matrix[i, :] > threshold]] = 1

        # shape: (len(indices, n)
        # if predict_columns[i, j] = 1, it means we should predict the quality of sevice j invoked by user i
        predict_columns = np.multiply(self.data[indices, :] == 0, self.raw_data[indices, :] > 0).astype(float)

        # shape(len(indices), self.num_of_services)
        # valid_counts[i, j] = 1 represents the quality value when user i invoke service j \
        # can be predicted by their similar users' quality value of invoking service j \
        # but it doesn't mean the quality value have to be predicted
        valid_counts = np.dot(similar_user_mat, (self.data > 0).astype(float))

        # to avoid divide by 0, set 0 to 1
        valid_counts[valid_counts == 0] = 1

        # shape:(len(indices), n)
        data_for_predicted = self.data.copy()
        data_for_predicted[data_for_predicted == -1] = 0

        # compute the average Qos value of user
        avg_value_of_user = np.average(data_for_predicted[indices, :], axis=1)
        # set the average value of users who haven't invoked any services to the average value of the whole matrix
        avg_value_of_user[avg_value_of_user == 0] = np.average(data_for_predicted)
        # extend the vector to a matrix with shape(len(indices), self.num_of_services)
        avg_value_of_user = np.tile(avg_value_of_user, (self.num_of_services, 1)).T

        predicted_value = np.dot(similar_user_mat, data_for_predicted)
        predicted_value[predicted_value == 0] = avg_value_of_user[predicted_value == 0]
        predicted_value = np.multiply(predict_columns, predicted_value)
        predicted_value = np.divide(predicted_value, valid_counts)
        total_predicted_count = np.sum(predicted_value > 0)

        #计算所有预测值的MAE和RMSE
        reference_data = (self.raw_data[indices, :])
        # reference_data[predicted_value == 0] = 0
        mae = np.sum(np.abs(predicted_value-reference_data))/total_predicted_count
        rmse = np.sum((predicted_value - reference_data) * (predicted_value - reference_data))
        rmse = np.sqrt(rmse/total_predicted_count)

        return mae, rmse, np.average(similar_user_mat)