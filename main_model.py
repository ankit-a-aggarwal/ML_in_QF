from sklearn.cluster import KMeans
from tensorflow import keras
from typing import List, Dict, Tuple
import os

import Calculate_Cluster_Weights
import RNN_Cell_Extensions
# external libs
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import Weight_And_Bias_Variables
from Returns_Vector_Wrapper import Returns_Vector_Linear_Wrapper
from CoVariance_Vector_Wrapper import Covariance_Matrix_Linear_Wrapper
import Cluster_Weights_Wrapper
class MainEncoderDecoder(models.Model):
    def __init__(self,
                 encoder_decoder_choice: str,
                 length_of_time_series: int,
                 embedding_dim: int,
                 num_layers: int = 2,
                 number_of_clusters: int = 8,
                 batch_size: int = 40,
                 hidden_layer_1: int = 100,
                hidden_layer_2: int = 50,
                hidden_layer_3: int = 50,
                 training=False) -> 'MainEncoderDecoder':
        super(MainEncoderDecoder, self).__init__()

        self.latent_dim = embedding_dim
        if training:
            self.denoising = True
        inputs = keras.Input(shape=(None,embedding_dim), name="Inputs")#,return_state=True
        self.bidirectional_layers = []
        GRU_layer_100_units =  layers.Bidirectional(layers.GRU(units=hidden_layer_1,return_sequences=True,return_state=True))
        GRU_layer_50_units_1 = layers.Bidirectional(layers.GRU(units=hidden_layer_2,recurrent_dropout=0.1,return_sequences=True,return_state=True))
        GRU_layer_50_units_2 = layers.Bidirectional(layers.GRU(units=hidden_layer_3,recurrent_dropout=0.2,return_sequences=True,return_state=True))
        self.bidirectional_layers.append(GRU_layer_100_units)
        self.bidirectional_layers.append(GRU_layer_50_units_1)
        self.bidirectional_layers.append(GRU_layer_50_units_2)
        #self.encoder = keras.Model(inputs,GRU_layer_50_units_2[0],name='Encoder')
        self.cell = keras.layers.GRUCell(2*(hidden_layer_1+hidden_layer_2+hidden_layer_3))
        self.cell2 = RNN_Cell_Extensions.LinearSpaceDecoderWrapper(self.cell,embedding_dim)
        self.decoder_layer = layers.RNN(self.cell2, return_sequences=True)
        self.lamda = 1
        initializer = keras.initializers.orthogonal(gain=1.0, seed=None)
        initial_value = initializer(shape=(batch_size,number_of_clusters))
        self.F = tf.Variable(initial_value,name='F',trainable=False)
        self.Returns_Vector = []
        self.Covariance_Matrix_Variables = []
        for i in range(batch_size):
            returns_wrapper = Returns_Vector_Linear_Wrapper()
            self.Returns_Vector.append(returns_wrapper)
        number_of_covariance_matrix_trainable_variables  = batch_size*(batch_size+1)//2
        for i in range(number_of_covariance_matrix_trainable_variables):
            covariance_matrix_wrapper = Covariance_Matrix_Linear_Wrapper()
            self.Covariance_Matrix_Variables.append(covariance_matrix_wrapper)

        self.weight1 = Weight_And_Bias_Variables.weight_variable(shape=[2*(hidden_layer_1+hidden_layer_2+hidden_layer_3), 64])
        self.bias1 = Weight_And_Bias_Variables.bias_variable(shape=[64])
        self.weight2 = Weight_And_Bias_Variables.weight_variable(shape=[64,2])
        self.bias2 = Weight_And_Bias_Variables.bias_variable(shape=[2])

        #print(tf.matmul(self.F.T,self.F))
        #autoencoder = keras.Model(inputs,decoder_layer,name='Autoencoder')



        #print(autoencoder.summary())



    def call(self,
             batch_inputs: Dict,
             epoch_number: int = None,
             training=False):
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        inputs : ``str``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, max_tokens_num) and entries are indices of tokens
            in to the vocabulary. 0 means that it's a padding token. max_tokens_num
            is maximum number of tokens in any text sequence in this batch.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.
        """
        out_of_sample_returns = batch_inputs['Validation_Inputs']
        inputs = batch_inputs['inputs']
        real_fake_data_labels = batch_inputs['real_fake_label']
        noise = batch_inputs['noise']
        noise_input = inputs
        if training:
            if self.denoising:
                print('Noise')
                noise_input = inputs + noise
            else:
                print('Non_noise')
                noise_input = inputs
        #encoded_hidden = self.encoder(noise_input)
        out = noise_input
        bidirectional_GRUs = self.bidirectional_layers
        hidden_state_list = []

        for index,layer in enumerate(bidirectional_GRUs):
            out = layer(out)
            out = out[0]
            hidden_state_list.append(out[:,out.shape[1]-1,:])
        encoded_hidden = tf.concat(hidden_state_list,axis=1)
        decoded = self.decoder_layer(noise_input,initial_state = encoded_hidden)
        hidden_abstract = encoded_hidden
        real_hidden_abstract = tf.split(hidden_abstract, 2)[0]
        # W has shape [sum(hidden_size)*2, batch_size]
        if 'F_Updated_Value' in batch_inputs:
            self.F = batch_inputs['F_Updated_Value']

        W = tf.transpose(real_hidden_abstract)#m*N
        WTW = tf.matmul(real_hidden_abstract, W)#[N*m][m*N]
        FTWTWF = tf.matmul(tf.matmul(tf.transpose(self.F), WTW), self.F) #[K*N][N*N][N*K]
        labels = tf.split(inputs, 2, axis=0)[0]
        predictions = tf.split(decoded, 2, axis=0)[0]
        reconstruction_loss = tf.keras.losses.mean_squared_error(labels, predictions)
        reconstruction_loss = tf.math.reduce_mean(reconstruction_loss)
        k_means_loss = tf.linalg.trace(WTW) - tf.linalg.trace(FTWTWF)

        hidden = tf.nn.relu(tf.matmul(hidden_abstract, self.weight1) + self.bias1) #[2N*400][400*256] + [256]
        output = tf.matmul(hidden, self.weight2) + self.bias2 #[2N*256] * [256*2] + [2]
        predict = tf.reshape(output, shape=[-1, 2])
        real_fake_data_labels = tf.stop_gradient(real_fake_data_labels)
        discriminative_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=real_fake_data_labels))
        total_loss = reconstruction_loss +  (self.lamda/2)*k_means_loss+discriminative_loss
        in_sample_adjusted_closed_returns = batch_inputs['Training_Inputs']
        out_of_sample_adjusted_closed_returns = batch_inputs['Validation_Inputs']
        print(reconstruction_loss.numpy(),k_means_loss.numpy(),discriminative_loss.numpy())
        if epoch_number % 2 ==0:
            estimation_loss,F_new,ticker_weights = self.KMeans(real_hidden_abstract,self.F.shape[1],
                                                in_sample_adjusted_closed_returns
                                                ,out_of_sample_adjusted_closed_returns,batch_inputs['batch_tickers'].numpy())
            total_loss += estimation_loss
            for i in range(self.F.shape[1]):
                print(i, ":", ticker_weights[i])
            return total_loss, F_new
        #print("Total loss ",total_loss)

        return total_loss,self.F

    def KMeans(self,hidden_state,number_of_clusters,in_sample_adjusted_closed_returns,out_of_sample_adjusted_closed_returns,
               tickers):
        part_hidden_val = np.array(hidden_state).reshape(-1, hidden_state.shape[1])  # np.sum(config.hidden_size) * 2
        W = part_hidden_val.T
        U, sigma, VT = np.linalg.svd(W)
        sorted_indices = np.argsort(sigma)
        topk_evecs = VT[sorted_indices[:-number_of_clusters - 1:-1], :]
        F_new = topk_evecs.T
        km = KMeans(n_clusters=number_of_clusters)
        stocks_in_each_cluster = {}
        for i in range(number_of_clusters):
            stocks_in_each_cluster[i] = []

        cluster_labels_training = km.fit_predict(X=part_hidden_val)
        for i in range(len(cluster_labels_training)):
            cluster_number = cluster_labels_training[i]
            stocks_in_each_cluster[cluster_number].append(tickers[i])
        for i in range(number_of_clusters):

            print(i,":",stocks_in_each_cluster[i])
        estimated_return_vector,estimated_covariance_matrix,cluster_weights,cluster_variance,ticker_weights = self.calculate_train_data_return_vector_for_all_stocks(cluster_labels_training,
                in_sample_adjusted_closed_returns,self.Returns_Vector,self.Covariance_Matrix_Variables,tickers)
        estimation_loss = self.calculate_estimation_loss_from_out_of_sample_returns(estimated_return_vector,cluster_weights,out_of_sample_adjusted_closed_returns)
        return estimation_loss,F_new,ticker_weights


    def calculate_estimation_loss_from_out_of_sample_returns(self,estimated_return_vector,
                cluster_weights,validation_period_out_of_sample_returns):
        asset_weights = [value for key, value in cluster_weights.items()]
        estimated_return_vector = tf.concat(estimated_return_vector, axis=0)
        mean_of_validation_period_out_of_sample_returns = tf.math.reduce_mean(
            validation_period_out_of_sample_returns, axis=1)
        difference_vector = estimated_return_vector - mean_of_validation_period_out_of_sample_returns
        difference_vector_1 = tf.math.pow(difference_vector, 2) # loss is per stock
        difference_vector_2 = tf.math.multiply(mean_of_validation_period_out_of_sample_returns,np.concatenate(asset_weights,axis=0))
        difference_vector_2 = -1* tf.math.reduce_sum(difference_vector_2)
        print(-1*difference_vector_2)
        return tf.math.reduce_mean(difference_vector_1)+difference_vector_2

    def calculate_train_data_return_vector_for_all_stocks(self,cluster_labels, returns_input,
                                                          estimated_return_trainable_variables,
                                                          estimated_covariance_matrix_trainable_variables,tickers):
        number_of_cluster = max(cluster_labels) + 1
        returns_cluster = {}
        cluster_of_covariance_matrix = {}
        ticker_returns_cluster = {}
        ticker_cluster_covariance_matrix = {}
        for i in range(number_of_cluster):
            returns_cluster[i] = []
            cluster_of_covariance_matrix[i] = []
            ticker_returns_cluster[i] = []
            ticker_cluster_covariance_matrix[i] = []
        for i in range(len(cluster_labels)):
            cluster_number = cluster_labels[i]
            returns_cluster[cluster_number].append(i)
            #ticker_cluster[cluster_number].append(tickers[i])
        estimated_return_vector = self.calculate_estimated_returns(estimated_return_trainable_variables, returns_input)
        estimated_returns_cluster = {}
        for i in range(number_of_cluster):
            estimated_returns_cluster[i] = []
        for key, value in returns_cluster.items():
            estimated_returns_cluster[key].append(tf.concat([estimated_return_vector[i] for i in value], axis=0))
            ticker_returns_cluster[key].append([[estimated_return_vector[i].numpy()[0][0],tickers[i]] for i in value])
        estimated_covariance_matrix = self.calculate_estimated_covariance_matrix(
            estimated_covariance_matrix_trainable_variables, returns_input, estimated_return_vector)
        for i in range(number_of_cluster):
            # cluster_number = cluster_labels[i]
            co_variance_matrix = estimated_covariance_matrix[np.ix_(returns_cluster[i]
                                               , returns_cluster[i])]
            cluster_tickers = tickers[returns_cluster[i]]
            co_variance_matrix = pd.DataFrame(co_variance_matrix,index=cluster_tickers,columns = cluster_tickers)
            cluster_of_covariance_matrix[i].append(estimated_covariance_matrix[np.ix_(returns_cluster[i]
                                                                                      , returns_cluster[i])])
            ticker_cluster_covariance_matrix[i] = co_variance_matrix
        cluster_weights, cluster_variance,ticker_weights = self.calculate_weight_vector_for_assets(ticker_returns_cluster,estimated_returns_cluster,
                                                                               cluster_of_covariance_matrix)
        # trainable_variables_loss = estimated_return_trainable_variables_loss + estimated_covariance_matrix_trainable_variables_loss
        # list_of_trainable_variables = estimated_return_list_of_trainable_variables+estimated_covariance_list_of_trainable_variables
        # list_of_trainable_variables.append(estimated_return_list_of_trainable_variables)
        # list_of_trainable_variables.append(estimated_covariance_list_of_trainable_variables)
        return estimated_return_vector, estimated_covariance_matrix, cluster_weights, cluster_variance,ticker_weights

        # print("")

    def calculate_estimated_returns(self,estimated_return_trainable_variables, returns_input):
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
        for index, j in enumerate(returns_input):
            wrapper = estimated_return_trainable_variables[index]

            return_estimate = wrapper(j)
            """
            for var in wrapper.trainable_variables:
                trainable_variables_loss += tf.math.add_n([tf.nn.l2_loss(var)])
                list_of_trainable_variables.append(var)
            """
            estimated_return_vector.append(return_estimate)
        return estimated_return_vector
        # ,trainable_variables_loss,list_of_trainable_variables

    def calculate_estimated_covariance_matrix(self,estimated_covariance_matrix_trainable_variables, returns_input,
                                              estimated_return_vector):

        estimated_covariance_matrix = np.empty((len(estimated_return_vector), len(estimated_return_vector)),
                                               dtype=np.float64)
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
            for k in range(j, len(returns_input)):
                returns_vec2 = returns_input[k]
                estimated_return_2 = estimated_return_vector[k]
                returns_vec3 = returns_vec - estimated_return_1
                returns_vec4 = returns_vec2 - estimated_return_2
                element_wise_return_product = tf.math.multiply(returns_vec3, returns_vec4)
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
        return estimated_covariance_matrix \
            # ,trainable_variables_loss,list_of_trainable_variables

    def calculate_weight_vector_for_assets(self,ticker_returns_cluster,estimated_returns_cluster, estimated_covariance_matrix):
        cluster_variance = np.ones(shape=len(estimated_covariance_matrix.keys()))
        cluster_weights = {}
        sum_of_cluster_weights = 0
        ticker_weights = {}
        for i in range(len(estimated_covariance_matrix.keys())):
            cluster_weights[i] = []
            ticker_weights[i] = []

        for key, value in estimated_returns_cluster.items():
            # cluster_weights_1 = get_inverse_variance_weights(value)
            cluster_weights_1,max_return_index = self.get_maximum_estimated_returns_momentum_based_weights(ticker_returns_cluster[key],value)
            cluster_weights_1 = cluster_weights_1 / len(estimated_returns_cluster.keys())
            cluster_weights[key] = cluster_weights_1
            return_ticker_value_list = ticker_returns_cluster[key]
            for i in range(len(return_ticker_value_list[0])):
                return_ticker_value_list[0][i].append(cluster_weights_1[i])
            ticker_weights[key] = return_ticker_value_list
            sum_of_cluster_weights += np.sum(cluster_weights_1)
        assert (sum_of_cluster_weights == 1)
        # return cluster_weights,cluster_variance

        for key, value in estimated_covariance_matrix.items():
            # cluster_weights_1 = get_inverse_variance_weights(value)
            """
            cluster_weights_1 = get_maximum_estimated_returns_momentum_based_weights(value)
            cluster_weights_1 = cluster_weights_1/len(estimated_covariance_matrix.keys())
            cluster_weights[key] = cluster_weights_1
            sum_of_cluster_weights += np.sum(cluster_weights_1)
            """
            cluster_weights_1 = cluster_weights[key]
            variance_of_cluster = self.calculate_variance(value, cluster_weights_1)
            cluster_variance[key] = variance_of_cluster
        # assert (sum_of_cluster_weights == 1)
        return cluster_weights, cluster_variance,ticker_weights

    def get_inverse_variance_weights(self,covariance):
        """
        Calculate inverse variance weight allocations.

        :param covariance: (pd.DataFrame) Covariance matrix of assets.
        :return: (np.array) Inverse variance weight values.
        """
        covariance = covariance[0]
        inv_diag = 1 / np.diag(covariance)
        parity_weights = inv_diag * (1 / np.sum(inv_diag))
        return parity_weights

    def get_maximum_estimated_returns_momentum_based_weights(self,ticker_return_tuple_list,value):
        weights = np.zeros(len(value[0]))
        max_returns_index = tf.math.argmax(value[0], axis=0)[0]
        weights[max_returns_index] = 1
        return weights,max_returns_index

    def calculate_variance(self,covariance, weights):
        """
        Calculate the variance of a portfolio.

        :param covariance: (pd.DataFrame/np.matrix) Covariance matrix of assets
        :param weights: (list) List of asset weights
        :return: (float) Variance of a portfolio
        """

        return np.dot(weights, np.dot(covariance[0], weights))











