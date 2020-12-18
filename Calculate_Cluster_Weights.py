
import numpy as np
import tensorflow as tf

from Returns_Vector_Wrapper import Returns_Vector_Linear_Wrapper
from CoVariance_Vector_Wrapper import Covariance_Matrix_Linear_Wrapper


def calculate_train_data_return_vector_for_all_stocks(cluster_labels,returns_input,
        estimated_return_trainable_variables,estimated_covariance_matrix_trainable_variables):
    number_of_cluster = max(cluster_labels)+1
    returns_cluster = {}
    cluster_of_covariance_matrix = {}

    for i in range(number_of_cluster):
        returns_cluster[i] = []
        cluster_of_covariance_matrix[i] = []
    for i in range(len(cluster_labels)):
        cluster_number = cluster_labels[i]
        returns_cluster[cluster_number].append(i)
    estimated_return_vector= calculate_estimated_returns(estimated_return_trainable_variables,returns_input)
    estimated_returns_cluster = {}
    for i in range(number_of_cluster):
        estimated_returns_cluster[i] = []
    for key,value in returns_cluster.items():
        estimated_returns_cluster[key].append(tf.concat([estimated_return_vector[i] for i in value],axis=0))
    estimated_covariance_matrix = calculate_estimated_covariance_matrix(estimated_covariance_matrix_trainable_variables,returns_input,estimated_return_vector)
    for i in range(number_of_cluster):
        #cluster_number = cluster_labels[i]
        cluster_of_covariance_matrix[i].append(estimated_covariance_matrix[np.ix_(returns_cluster[i]
        ,returns_cluster[i])])

    cluster_weights,cluster_variance = calculate_weight_vector_for_assets(estimated_returns_cluster,cluster_of_covariance_matrix)
    #trainable_variables_loss = estimated_return_trainable_variables_loss + estimated_covariance_matrix_trainable_variables_loss
    #list_of_trainable_variables = estimated_return_list_of_trainable_variables+estimated_covariance_list_of_trainable_variables
    #list_of_trainable_variables.append(estimated_return_list_of_trainable_variables)
    #list_of_trainable_variables.append(estimated_covariance_list_of_trainable_variables)
    return estimated_return_vector,estimated_covariance_matrix,cluster_weights,cluster_variance

    #print("")


def calculate_estimated_returns(estimated_return_trainable_variables,returns_input):
    estimated_return_vector = []
    """ 
    estimated_returns_vector = {}
    keys = returns_cluster.keys()
    for i in keys:
        estimated_returns_vector[i] = []
    for key,values in returns_cluster.items():
        for j in values:
            wrapper = Returns_Vector_Linear_Wrapper(j)
            return_estimate = wrapper(j)
            estimated_returns_vector[key].append(return_estimate)
    """
    trainable_variables_loss = 0
    list_of_trainable_variables = []
    for index,j in enumerate(returns_input):
        wrapper = estimated_return_trainable_variables[index]

        return_estimate = wrapper(j)
        """
        for var in wrapper.trainable_variables:
            trainable_variables_loss += tf.math.add_n([tf.nn.l2_loss(var)])
            list_of_trainable_variables.append(var)
        """
        estimated_return_vector.append(return_estimate)
    return estimated_return_vector
    #,trainable_variables_loss,list_of_trainable_variables

def calculate_estimated_covariance_matrix(estimated_covariance_matrix_trainable_variables,returns_input,estimated_return_vector):

    estimated_covariance_matrix = np.empty((len(estimated_return_vector),len(estimated_return_vector)),dtype=np.float64)
    """
    keys = returns_cluster.keys()
    keys = returns_cluster.keys()
    for i in keys:
        cluster_covariance_matrix[i] = []
    for key,value in returns_cluster.items():
        cluster_covariance_matrix[key] = np.empty((len(value),len(value)),dtype=np.float64)
    """
    trainable_variables_loss = 0
    list_of_trainable_variables = []
    l = 0
    for j in range(len(returns_input)):
        returns_vec = returns_input[j]
        estimated_return_1 = estimated_return_vector[j]
        for k in range(j,len(returns_input)):
            returns_vec2 = returns_input[k]
            estimated_return_2 = estimated_return_vector[k]
            returns_vec3 = returns_vec - estimated_return_1
            returns_vec4 = returns_vec2 - estimated_return_2
            element_wise_return_product = tf.math.multiply(returns_vec3,returns_vec4)
            wrapper = estimated_covariance_matrix_trainable_variables[l]
            estimated_covariance_value = wrapper(element_wise_return_product)
            l += 1
            """
            for var in wrapper.trainable_variables:
                trainable_variables_loss += tf.math.add_n([tf.nn.l2_loss(var)])
                list_of_trainable_variables.append(var)
            """
            estimated_covariance_matrix[j][k] = estimated_covariance_value
            estimated_covariance_matrix[k][j] = estimated_covariance_value
    return estimated_covariance_matrix\
        #,trainable_variables_loss,list_of_trainable_variables


def calculate_weight_vector_for_assets(estimated_returns_cluster,estimated_covariance_matrix):
    cluster_variance = np.ones(shape=len(estimated_covariance_matrix.keys()))
    cluster_weights = {}
    sum_of_cluster_weights = 0
    for i in range(len(estimated_covariance_matrix.keys())):
        cluster_weights[i] = []

    for key,value in estimated_returns_cluster.items():
        #cluster_weights_1 = get_inverse_variance_weights(value)
        cluster_weights_1 = get_maximum_estimated_returns_momentum_based_weights(value)
        cluster_weights_1 = cluster_weights_1/len(estimated_returns_cluster.keys())
        cluster_weights[key] = cluster_weights_1
        sum_of_cluster_weights += np.sum(cluster_weights_1)
    assert (sum_of_cluster_weights == 1)
    #return cluster_weights,cluster_variance

    for key,value in estimated_covariance_matrix.items():
        #cluster_weights_1 = get_inverse_variance_weights(value)
        """
        cluster_weights_1 = get_maximum_estimated_returns_momentum_based_weights(value)
        cluster_weights_1 = cluster_weights_1/len(estimated_covariance_matrix.keys())
        cluster_weights[key] = cluster_weights_1
        sum_of_cluster_weights += np.sum(cluster_weights_1)
        """
        cluster_weights_1 = cluster_weights[key]
        variance_of_cluster = calculate_variance(value,cluster_weights_1)
        cluster_variance[key] = variance_of_cluster
    #assert (sum_of_cluster_weights == 1)
    return cluster_weights,cluster_variance


def get_inverse_variance_weights(covariance):
    """
    Calculate inverse variance weight allocations.

    :param covariance: (pd.DataFrame) Covariance matrix of assets.
    :return: (np.array) Inverse variance weight values.
    """
    covariance = covariance[0]
    inv_diag = 1 / np.diag(covariance)
    parity_weights = inv_diag * (1 / np.sum(inv_diag))
    return parity_weights


def get_maximum_estimated_returns_momentum_based_weights(value):
    weights = np.zeros(len(value))
    max_returns_index = tf.math.argmax(value,axis=0)
    weights[max_returns_index] = 1
    return weights



def calculate_variance(covariance, weights):
    """
    Calculate the variance of a portfolio.

    :param covariance: (pd.DataFrame/np.matrix) Covariance matrix of assets
    :param weights: (list) List of asset weights
    :return: (float) Variance of a portfolio
    """

    return np.dot(weights, np.dot(covariance[0], weights))







