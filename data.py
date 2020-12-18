from collections import Counter
from typing import List, Dict, Tuple, Any
import json
import os
import zipfile

# external lib imports:
import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import copy
from sklearn.preprocessing import minmax_scale

def read_instances(data_file_path: str,train_test_or_validation: str) -> List[pd.DataFrame]:

    data = pd.read_csv(data_file_path)
    unique_tickers = data['Ticker'].unique()
    data = data.sort_values(['Date'])
    data = data[['Date','Ticker','AdjustedClose','Adj. Volume','Returns_from_adjusted_closed_to_adjusted_closed']]
    num_features = len(data.columns)
    list_of_dataframes = []
    max_length = -1
    count = 0
    number_of_columns = 0
    unique_tickers = unique_tickers[:80]
    for ticker in tqdm(unique_tickers):
        ticker_data = data[data['Ticker'] == ticker]
        scaled_adjusted_closed = minmax_scale(ticker_data.loc[:,'AdjustedClose'])
        ticker_data.loc[:,'AdjustedClose'] = scaled_adjusted_closed
        #if len(ticker_data) != max_length:
            #continue
        #ticker_data = ticker_data.loc[:,ticker_data.columns !='Ticker']
        ticker_data = ticker_data.loc[ticker_data['Returns_from_adjusted_closed_to_adjusted_closed'].notnull(),:]
        ticker_data = ticker_data.loc[:,ticker_data.columns !='Adj. Volume']
        number_of_columns = len(ticker_data.columns)
        """
        if train_test_or_validation == 'train':
            if len(ticker_data) == 1366:
                #print(ticker_data['Ticker'])
                count+=1
        elif train_test_or_validation == 'validation':

            if len(ticker_data) == 585:
                #print(ticker_data['Ticker'])
                count+=1
        """
        ticker_data = ticker_data[:550]
        max_length = max(len(ticker_data),max_length)
        #x_shaped = ticker_data.T
        #print(x_shaped.shape)
        list_of_dataframes.append(ticker_data)
    #batch_data = np.stack(list_of_dataframes)
    #sliced_training_data = data[['Date','Ticker','Returns_hat','Returns_Standard_Deviation','Adj. Volume']]
    #sliced_training_data = sliced_training_data.loc[sliced_training_data['Returns_hat'].notnull(),:]
    #print(count)
    #assert (count == len(unique_tickers))
    return list_of_dataframes,max_length,number_of_columns


def generate_batches(instances1: List[pd.DataFrame], batch_size,length_of_time_series,embedding_dim) -> List[Dict[str, np.ndarray]]:
    """
    Generates and returns batch of tensorized instances in a chunk of batch_size.
    """

    def chunk(items: List[Any], num: int):
        return [items[index:index+num] for index in range(0, len(items), num)]
    instances = copy.deepcopy(instances1)
    #np.random.shuffle(instances)
    batches = []
    num_row_ids = [len(instance) for instance in instances]
    #len(instances[0])
    max_num_row_ids = max(num_row_ids)

    Ticker_list = [instance.Ticker.values[0] for instance in instances]
    Date_list = instances[0].Date.values
    asset_returns = np.zeros((len(instances), max_num_row_ids,1), dtype=np.float32)
    for index,instance in enumerate(instances):
        instance = instance.loc[:, instance.columns != 'Ticker']
        instance = instance.loc[:, instance.columns != 'Date']
        asset_returns[index] = instance.loc[:,instance.columns == 'Returns_from_adjusted_closed_to_adjusted_closed']
        instance = instance.loc[:, instance.columns != 'Returns_from_adjusted_closed_to_adjusted_closed']
        instances[index] = instance
        #instance.drop('Ticker',axis=1,inplace=True)
        #instance.drop('Date',axis=1,inplace=True)
    num_token_ids = [len(instance.columns) for instance in instances]
    max_num_token_ids = max(num_token_ids)
    batch = {"inputs":np.zeros((len(instances), max_num_row_ids,max_num_token_ids), dtype=np.float32)}
    batch['inputs'] = np.stack(instances)
    batch['Ticker'] = np.array(Ticker_list)
    batch['Date'] = np.array(Date_list)
    batch['Asset_Returns'] = asset_returns
    batches.append(batch)
    #batches_of_instances = chunk(instances, batch_size)
    """
    batches = []
    for batch_of_instances in tqdm(batches_of_instances):

        
        num_row_ids = [len(instance)
                         for instance in batch_of_instances]
        max_num_row_ids = max(num_row_ids)
        count = min(batch_size, len(batch_of_instances))
        #indices = np.random.permutation(count)
        #batch_of_instances = np.stack(batch_of_instances)

        #batch_of_instances = tf.convert_to_tensor(batch_of_instances)
        batch = {"inputs": np.zeros((count, max_num_row_ids,max_num_token_ids), dtype=np.float32)}        #if "labels" in  batch_of_instances[0]:
         #   batch["labels"] = np.zeros(count, dtype=np.int32)
        #list_of_instances = []
        if "Ticker" in batch_of_instances[0]:
            batch["Ticker"] = np.ndarray(count, dtype=object)
        if "Date" in batches_of_instances[0]:
            batch["Date"] = np.ndarray(max_num_row_ids, dtype=object)

        for batch_index, instance in enumerate(batch_of_instances):
            #num_tokens = len(instance.columns)
            #inputs = np.array(instance.columns)
            #num_rows = len(instance)
            #inputs = instance.T
            #list_of_instances.append(instance)
            batch['inputs'][batch_index] = instance.loc[:,instance.columns !='Ticker']
            batch['Ticker'][batch_index] = instance['Ticker'].values[0]
            batch['Date'][batch_index] = instance['Date'].values
        #input = np.stack(list_of_instances)
        #batch["inputs"] = input

            #if "labels" in instance:
                #batch["labels"][batch_index] = np.array(instance["labels"])
        batches.append(batch)
    """
    return batches,max_num_token_ids
    #return batches

def get_fake_sample(data):
    sample_nums = data.shape[0]
    series_len = data.shape[1]
    embedding_size = data.shape[2]
    mask = np.ones(shape=[sample_nums, series_len,embedding_size])
    rand_list = np.zeros(shape=[sample_nums, series_len,embedding_size])

    fake_position_nums = int(series_len * 0.2)
    fake_position = np.random.randint(low=0, high=series_len, size=[sample_nums, fake_position_nums])

    for i in range(fake_position.shape[0]):
        for j in range(fake_position.shape[1]):
            mask[i, fake_position[i, j]] = 0

    for i in range(rand_list.shape[0]):
        count = 0
        for j in range(rand_list.shape[1]):
            if j in fake_position[i]:
                assign_data_to_random_index = data[i, fake_position[i, count]]
                rand_list[i, j] = assign_data_to_random_index
                count += 1
    fake_data = data * mask + rand_list * (1 - mask)
    real_fake_labels = np.zeros(shape=[sample_nums * 2, 2])
    for i in range(sample_nums * 2):
        if i < sample_nums:
            real_fake_labels[i, 0] = 1
        else:
            real_fake_labels[i, 1] = 1
    return fake_data, real_fake_labels


